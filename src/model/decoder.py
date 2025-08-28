import torch
import torch.nn as nn


TRANSFORM_TOKENS = {
    "[PAD]": 0,        # Padding token (must be 0 for proper masking)
    "[START]": 1,      # Start token
    "[END]": 2,        # End token
    "[NOOP]": 3,       # No operation (identity transformation)
    "grayscale": 4,
    "rotate_90": 5,
    "rotate_180": 6,
    "rotate_270": 7,
    "color_jitter": 8,
    "noise_adding": 9,
    "crop": 10,
    "horizontal_flip": 11,
    "vertical_flip": 12,
}

class TransformEmbedding(nn.Module):
    """
    Embedding layer for transformation tokens.
    Converts token IDs to dense vector representations.
    """
    def __init__(self, vocab_size, embedding_dim, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, tokens):
        """
        Args:
            tokens: Input token IDs [batch_size, seq_len]

        Returns:
            torch.Tensor: Embeddings for input tokens [batch_size, seq_len, embedding_dim]
        """
        return self.embedding(tokens)

class TransformDecoder(nn.Module):
    """
    Transformer-based decoder for predicting sequences of image transformations.
    Takes image features and generates transformation sequences.
    """
    def __init__(self, image_feature_dim, embedding_dim, num_heads, num_layers, vocab_size, max_seq_length=20):
        """
        Args:
            image_feature_dim: Dimension of input image features
            embedding_dim: Dimension for token embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            vocab_size: Size of transformation vocabulary
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.max_seq_length = max_seq_length

        # Linear projection for image features
        self.image_projection = nn.Linear(image_feature_dim, embedding_dim)

        # Token embeddings with padding index 0
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(embedding_dim, num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Linear layer for token prediction
        self.token_predictor = nn.Linear(embedding_dim, vocab_size)

    def create_attention_mask(self, input_tokens):
        """
        Create attention mask to ignore padding tokens.

        Args:
            input_tokens: Input token IDs [batch_size, seq_len]

        Returns:
            torch.Tensor: Attention mask [batch_size, seq_len]
        """
        return (input_tokens == 0)  # Mask for padding tokens (0 is the index for [PAD])

    def forward(self, image_features, target_tokens=None):
        """
        Args:
            image_features: Image features [batch_size, image_feature_dim]
            target_tokens: Target transformation tokens [batch_size, seq_len] (optional)

        Returns:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            or generated sequence if target_tokens is None
        """
        batch_size = image_features.size(0)

        # Project image features to embedding dimension
        memory = self.image_projection(image_features).unsqueeze(0)  # [1, batch_size, embedding_dim]

        if target_tokens is None:
            # Generation mode
            input_tokens = torch.full((batch_size, 1), TRANSFORM_TOKENS["[START]"],
                                     device=image_features.device, dtype=torch.long)

            generated_tokens = []
            for step in range(self.max_seq_length):
                token_embeddings = self.token_embedding(input_tokens)
                positions = torch.arange(input_tokens.size(1), device=image_features.device).unsqueeze(0)
                token_embeddings += self.position_embedding(positions)

                decoder_output = self.transformer_decoder(
                    token_embeddings.transpose(0, 1), memory)

                logits = self.token_predictor(decoder_output[-1])
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated_tokens.append(next_token)

                if (next_token == TRANSFORM_TOKENS["[END]"]).all():
                    break

                input_tokens = torch.cat([input_tokens, next_token], dim=1)

            return torch.cat(generated_tokens, dim=1)
        else:
            # Training mode with target tokens
            token_embeddings = self.token_embedding(target_tokens)
            positions = torch.arange(target_tokens.size(1), device=image_features.device).unsqueeze(0)
            token_embeddings += self.position_embedding(positions)

            tgt_mask = self.create_attention_mask(target_tokens)

            decoder_output = self.transformer_decoder(
                token_embeddings.transpose(0, 1), memory,
                tgt_key_padding_mask=tgt_mask)

            logits = self.token_predictor(decoder_output)

            return logits
