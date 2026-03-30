import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from omegaconf import DictConfig

from .vit_encoder import ImagePairEncoderViT
from .decoder import TransformDecoder


class QueryCompressorLayer(nn.Module):
    """
    Один слой Q-Former-подобной компрессии:
    - обучаемые query-токены смотрят cross-attention-ом на полную карту патчей.
    """

    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, M, D] — текущие query-токены.
            tokens:  [B, L, D] — полный выход ViT (CLS+патчи после proj).
        Returns:
            queries_out: [B, M, D]
        """
        x = queries
        q = self.ln1(x)
        k = self.ln1(tokens)
        v = k
        attn_out, _ = self.attn(query=q, key=k, value=v, need_weights=False)
        x = x + self.dropout(attn_out)

        y = self.ln2(x)
        y = self.mlp(y)
        x = x + self.dropout(y)
        return x


class QueryCompressor(nn.Module):
    """
    Небольшой Q-Former-подобный блок для сжатия 197 токенов ViT в M "смарт"-токенов.

    - Вход:  [B, 197, D]  (CLS + патчи после proj).
    - Выход: [B, M,   D]  (compressed tokens).
    """

    def __init__(
        self,
        dim: int,
        num_queries: int = 32,
        num_layers: int = 2,
        n_head: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dim))
        self.layers = nn.ModuleList(
            [QueryCompressorLayer(dim=dim, n_head=n_head, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, L, D] (ожидается L=197, но можно и другое L).
        Returns:
            compressed: [B, M, D]
        """
        B, _, D = tokens.shape
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"

        queries = self.query_tokens.expand(B, -1, -1)  # [B, M, D]
        for layer in self.layers:
            queries = layer(queries, tokens)
        return queries


class ImagePairEncoderViTQFormer(ImagePairEncoderViT):
    """
    Вариант энкодера, который поверх существующего ViT+proj добавляет Q-Former-подобный
    модуль с learnable queries для сжатия карты патчей до фиксированного числа токенов.

    - ViT и proj полностью совместимы по именам и могут загружать уже обученные веса.
    - Обычно ViT и proj замораживаются, обучается только QueryCompressor (и, при желании,
      декодер).
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        num_queries = getattr(config, "num_queries", 32)
        num_layers = getattr(config, "num_qformer_layers", 2)
        n_head = getattr(config, "qformer_n_head", 8)
        dropout = getattr(config, "qformer_dropout", float(getattr(config, "dropout", 0.1)))

        self.query_compressor = QueryCompressor(
            dim=self.output_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            n_head=n_head,
            dropout=dropout,
        )

    def extract_image_embeddings(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Кодирует две картинки и возвращает уже сжатые последовательности:
            [B, M, out_token_n_embd] для каждой.
        """
        tokens1 = self.image_encoder(image_batch_1)  # [B, 197, D_vit]
        tokens2 = self.image_encoder(image_batch_2)  # [B, 197, D_vit]

        proj1 = self.proj(tokens1)  # [B, 197, E]
        proj2 = self.proj(tokens2)  # [B, 197, E]

        comp1 = self.query_compressor(proj1)  # [B, M, E]
        comp2 = self.query_compressor(proj2)  # [B, M, E]
        return comp1, comp2

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        use_precomputed_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Возвращает объединённую последовательность сжатых токенов:
            [B, M1 + 1 + M2, out_token_n_embd].

        Если use_precomputed_embeddings=True, ожидает уже сжатые эмбеддинги
        формы [B, M, E] на входе и просто добавляет sep_token.
        """
        if not use_precomputed_embeddings:
            comp1, comp2 = self.extract_image_embeddings(image_batch_1, image_batch_2)
        else:
            comp1, comp2 = image_batch_1, image_batch_2

        B = comp1.shape[0]
        sep = self.sep_token.expand(B, 1, -1)  # [B, 1, E]
        combined = torch.cat([comp1, sep, comp2], dim=1)
        return combined


class ImageTransformPredictorViTQFormer(nn.Module):
    """
    Полная модель, аналогичная ImageTransformPredictor, но:
      - вместо ImagePairEncoderEfficientNet / ImagePairEncoderViT использует
        ImagePairEncoderViTQFormer;
      - сжатое число токенов (M << 197) подаётся в существующий TransformDecoder.

    При использовании можно загрузить веса ViT+decoder из уже обученной ViT-модели,
    а затем доучивать только QueryCompressor (и, по желанию, часть декодера).
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        encoder_cfg = config.encoder
        encoder_type = encoder_cfg.get("type", "vit")
        if encoder_type != "vit":
            raise ValueError(
                f"ImageTransformPredictorViTQFormer рассчитан на encoder.type='vit', "
                f"получено encoder.type={encoder_type!r}"
            )

        self.image_pair_encoder = ImagePairEncoderViTQFormer(encoder_cfg)

        self.bos_token_id = config.decoder.bos_token_id
        self.eos_token_id = config.decoder.eos_token_id
        self.pad_token_id = config.decoder.pad_token_id

        self.transform_decoder = TransformDecoder(config.decoder)

        freeze_decoder = getattr(config, "freeze_decoder", False)
        if freeze_decoder:
            for p in self.transform_decoder.parameters():
                p.requires_grad = False

    def extract_image_embeddings(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.image_pair_encoder.extract_image_embeddings(image_batch_1, image_batch_2)

    def forward(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        idx: torch.LongTensor,
        use_precomputed_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images_embeddings = self.image_pair_encoder(
            image_batch_1,
            image_batch_2,
            use_precomputed_embeddings=use_precomputed_embeddings,
        )

        targets = torch.roll(idx, shifts=-1, dims=1)
        targets[:, -1] = self.pad_token_id

        logits, loss = self.transform_decoder(
            idx=idx,
            images_embeddings=images_embeddings,
            targets=targets,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        if max_new_tokens is None:
            max_new_tokens = self.config.decoder.max_seq_len - 1

        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if bos_token_id is None:
            bos_token_id = self.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        images_embeddings = self.image_pair_encoder(image_batch_1, image_batch_2)

        return self.transform_decoder.generate(
            images_embeddings=images_embeddings,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

    @torch.no_grad()
    def generate_step_with_cross_attn(
        self,
        image_batch_1: torch.Tensor,
        image_batch_2: torch.Tensor,
        idx_prefix: Optional[torch.Tensor] = None,
        use_precomputed_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if idx_prefix is None:
            device = image_batch_1.device
            B = image_batch_1.shape[0]
            idx_prefix = torch.full(
                (B, 1),
                self.bos_token_id,
                dtype=torch.long,
                device=device,
            )

        images_embeddings = self.image_pair_encoder(
            image_batch_1,
            image_batch_2,
            use_precomputed_embeddings=use_precomputed_embeddings,
        )

        return self.transform_decoder.generate_step_with_cross_attn(
            images_embeddings=images_embeddings,
            idx_prefix=idx_prefix,
        )

