import torch
import torch.nn as nn
from .encoder import ImagePairEncoder
from .decoder import TransformDecoder
from transformers import PreTrainedModel, GPT2Config
from typing import Optional, Union
from .tokenizer import TRANSFORM_TOKENS, VOCAB_SIZE, START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID


class ImageTransformPredictor(PreTrainedModel):
    """
    Complete end-to-end model for predicting image transformation sequences from image pairs.

    Architecture:
        1. ImagePairEncoder: Extracts fused features from two images using EfficientNet-B3
        2. TransformDecoder: Autoregressively generates sequence of transformation tokens
           (including "noop", "crop", "rotate_90", etc.) conditioned on the fused embedding

    Input: Two images [B, 3, 224, 224]
    Output: Sequence of token IDs [B, L] — e.g., ["noop"], ["crop", "grayscale"], ["resize", "crop"]

    Special Tokens:
        [PAD] = 0   --> padding
        [START] = 1 --> start of sequence
        [END] = 2   --> end of sequence
        "noop" = 3  --> identity transform (no change)

    Usage:
        model = ImageTransformPredictor(embedding_dim=512)
        logits = model(img1, img2, target_tokens)  # training
        generated = model.generate(img1, img2)     # inference
    """

    config_class = GPT2Config

    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        num_layers: int = 3,
        max_seq_length: int = 10,
        freeze_image_encoder: bool = True,
        unfreeze_n_layers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_embd=embedding_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_seq_length,
            n_ctx=max_seq_length,
            bos_token_id=START_TOKEN_ID,
            eos_token_id=END_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            use_cache=True,
        ))

        # 1. Image pair encoder (EfficientNet-B3 based)
        self.image_pair_encoder = ImagePairEncoder(
            embedding_dim=embedding_dim,
            freeze_image_encoder=freeze_image_encoder,
            unfreeze_n_layers=unfreeze_n_layers,
        )

        # 2. Transformation decoder (Transformer-based autoregressive generator)
        self.transform_decoder = TransformDecoder(self.config)

        # Store config and parameters for saving/loading
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.freeze_image_encoder = freeze_image_encoder
        self.unfreeze_n_layers = unfreeze_n_layers

        # Initialize weights (decoder already initialized via post_init())
        self.init_weights()

    def init_weights(self):
        """Initialize fuser layer if needed — already done in ImagePairEncoder"""
        pass  # Decoder uses its own init, encoder is initialized in its constructor

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        target_tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass during training.

        Args:
            image_batch_1: [B, 3, 224, 224] — source image batch
            image_batch_2: [B, 3, 224, 224] — target image batch
            target_tokens: [B, L] — ground truth token sequence (with [START] and [END])
            attention_mask: [B, L] — optional mask for padding (if not provided, auto-generated)

        Returns:
            logits: [B, L, V] — logits for next-token prediction
        """
        # Encode image pair --> [B, embedding_dim]
        image_features = self.image_pair_encoder(image_batch_1, image_batch_2)

        if target_tokens is None:
            raise ValueError("target_tokens must be provided during training.")

        # Generate input_ids: [START] + target_tokens[:-1]
        start_token = torch.full((target_tokens.size(0), 1), START_TOKEN_ID, dtype=torch.long, device=target_tokens.device)
        input_ids = torch.cat([start_token, target_tokens[:, :-1]], dim=1)  # [B, L]

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != PAD_TOKEN_ID).long()

        # Pass through decoder
        outputs = self.transform_decoder(
            image_embeddings=image_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_tokens,
        )

        return outputs.logits  # Return logits for loss computation

    def generate(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        early_stopping: bool = True,
    ) -> torch.LongTensor:
        """
        Autoregressive generation of transformation sequence from image pair.

        Args:
            image_batch_1: [B, 3, 224, 224] — source image
            image_batch_2: [B, 3, 224, 224] — target image
            max_length: Max tokens to generate (defaults to model's max_seq_length)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: If False, greedy decoding
            early_stopping: Stop when all sequences hit [END]

        Returns:
            Generated token IDs: [B, generated_length]
        """
        if max_length is None:
            max_length = self.max_seq_length

        # Encode image pair
        image_features = self.image_pair_encoder(image_batch_1, image_batch_2)  # [B, D]

        # Generate using decoder
        return self.transform_decoder.generate(
            image_embeddings=image_features,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            early_stopping=early_stopping,
        )

    def save_pretrained(self, save_directory: str):
        """
        Save full model: encoder + decoder + config.
        Compatible with Hugging Face ecosystem.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save decoder (HF-compatible)
        self.transform_decoder.save_pretrained(os.path.join(save_directory, "decoder"))

        # Save encoder state dict
        torch.save(self.image_pair_encoder.state_dict(), os.path.join(save_directory, "encoder.pth"))

        # Save config
        config = {
            "embedding_dim": self.embedding_dim,
            "num_heads": self.transform_decoder.config.n_head,
            "dim_feedforward": self.transform_decoder.config.n_inner,
            "num_layers": self.transform_decoder.config.n_layer,
            "max_seq_length": self.max_seq_length,
            "freeze_image_encoder": self.freeze_image_encoder,
            "unfreeze_n_layers": self.unfreeze_n_layers,
        }
        import json
        with open(os.path.join(save_directory, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """
        Load full model from saved directory.
        """
        import json
        with open(os.path.join(load_directory, "config.json"), 'r') as f:
            config = json.load(f)

        model = cls(
            embedding_dim=config["embedding_dim"],
            num_heads=config["num_heads"],
            dim_feedforward=config["dim_feedforward"],
            num_layers=config["num_layers"],
            max_seq_length=config["max_seq_length"],
            freeze_image_encoder=config["freeze_image_encoder"],
            unfreeze_n_layers=config["unfreeze_n_layers"],
        )

        # Load decoder
        model.transform_decoder = TransformDecoder.from_pretrained(
            os.path.join(load_directory, "decoder")
        )

        # Load encoder
        model.image_pair_encoder.load_state_dict(
            torch.load(os.path.join(load_directory, "encoder.pth"), map_location="cpu")
        )

        return model