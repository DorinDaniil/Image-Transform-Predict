import torch
import torch.nn as nn
import numpy as np
from typing import Union, List
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from efficientnet_pytorch import EfficientNet
import io


preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])



class PlagiarismDetectionModel(nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device] = None
    ):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = ImageEncoder()
        self.fuser = ImageFuser()
        self.decoder = TransformationDecoder()
        
        self.to(self.device)

    def load_components(
        self,
        encoder_path: str,
        fuser_path: str,
        decoder_path: str,
        strict: bool = True
    ) -> None:

        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=self.device, weights_only=True),
            strict=strict
        )
        self.fuser.load_state_dict(
            torch.load(fuser_path, map_location=self.device, weights_only=True),
            strict=strict
        )
        self.decoder.load_state_dict(
            torch.load(decoder_path, map_location=self.device, weights_only=True),
            strict=strict
        )

    def preprocess_batch(
        self,
        images: List[Union[bytes, Image.Image, np.ndarray, torch.Tensor]]
    ) -> torch.Tensor:

        processed = []
        for img in images:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, torch.Tensor):
                if img.ndim == 3 and img.shape[-1] in [1, 3]:  # HWC
                    img_np = img.cpu().numpy()
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    img = Image.fromarray(img_np)
                else:
                    img = transforms.ToPILImage()(img)
            
            processed.append(preprocess(img))
        
        return torch.stack(processed)  # [B, C, H, W]

    def forward(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:

        x1 = images1.to(self.device)
        x2 = images2.to(self.device)
        
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)

        fused = self.fuser(emb1, emb2)
        
        B = fused.shape[0]
        idx_tokens = torch.full(
            (B, 1), 
            self.decoder.bos_token_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        logits = self.decoder(fused, idx_tokens)
        logits[:, self.decoder.pad_token_id] = -float('inf')
        logits[:, self.decoder.bos_token_id] = -float('inf')
    
        probs = F.softmax(logits, dim=-1)  # Стабильная реализация
        
        p_eos = probs[:, self.decoder.eos_token_id]
        similarities = 1.0 - p_eos
        
        return similarities

    @torch.inference_mode()  # Чуть быстрее, чем no_grad
    def compute_similarities_batch(
        self,
        query_images: List[Union[bytes, np.ndarray, torch.Tensor, Image.Image]],
        candidate_images: List[List[Union[bytes, np.ndarray, torch.Tensor, Image.Image]]],
        temperature: float = 1.0,
        candidate_chunk_size: int = 256,  # Увеличили чанк
        preprocess_batch_size: int = 64   # Новый параметр: пакетная предобработка
    ) -> List[List[float]]:
        if not query_images:
            return []
        
        B = len(query_images)
        results = []
        
        # 1. Предобработка запросов (один раз)
        query_batch = self.preprocess_batch(query_images)  # [B, 3, 300, 300]
        query_embs = self.encoder(query_batch.to(self.device))  # [B, d]
        
        for i in range(B):
            candidates_i = candidate_images[i]
            if not candidates_i:
                results.append([])
                continue
            
            # 2. ПАКЕТНАЯ предобработка кандидатов (ключевое ускорение!)
            cand_embs_list = []
            for start in range(0, len(candidates_i), preprocess_batch_size):
                chunk = candidates_i[start:start + preprocess_batch_size]
                # Один вызов preprocess_batch на весь чанк вместо вызова на каждый элемент
                preprocessed = self.preprocess_batch(chunk)  # [N, 3, 300, 300]
                with torch.inference_mode():
                    embs = self.encoder(preprocessed.to(self.device))  # [N, d]
                cand_embs_list.append(embs)
            
            all_cand_embs = torch.cat(cand_embs_list, dim=0)  # [N_total, d]
            N_cand = all_cand_embs.shape[0]
            
            similarities = []
            
            # 3. Инференс декодера чанками
            q_emb = query_embs[i:i+1]  # [1, d]
            
            for start in range(0, N_cand, candidate_chunk_size):
                end = min(start + candidate_chunk_size, N_cand)
                cand_embs_chunk = all_cand_embs[start:end]  # [C, d]
                C = cand_embs_chunk.shape[0]
                
                # Расширяем эмбеддинг запроса под размер чанка
                q_expanded = q_emb.expand(C, -1)  # [C, d]
                
                fused = self.fuser(cand_embs_chunk, q_expanded)  # [C, 1, embed_dim]
                idx_tokens = torch.full(
                    (C, 1), 
                    self.decoder.bos_token_id, 
                    dtype=torch.long, 
                    device=self.device
                )
                logits = self.decoder(images_embeddings=fused, idx=idx_tokens)  # [C, V]
                
                # Маскировка + softmax + извлечение EOS
                logits[:, self.decoder.pad_token_id] = -float('inf')
                logits[:, self.decoder.bos_token_id] = -float('inf')
                p_eos = F.softmax(logits, dim=-1)[:, self.decoder.eos_token_id]  # [C]
                
                similarities.extend((1.0 - p_eos).cpu().tolist())
                
                # Убираем empty_cache — PyTorch сам управляет памятью в inference_mode
                # Если память заканчивается — добавьте условную очистку:
                # if C == candidate_chunk_size and torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
                #     torch.cuda.empty_cache()
            
            results.append(similarities)
        
        return results

# =============================================================================
# ImageEncoder:
# =============================================================================

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = EfficientNet.from_name('efficientnet-b3')
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()

        self.feature_dim = 1536

    def forward(self, x):
        features = self.backbone(x)  # [batch_size, 1536]
        return features


# =============================================================================
# ImageFuser:
# =============================================================================

class ImageFuser(nn.Module):
    def __init__(self, embed_dim: int = 512, concat_embeds_dim: int = 3072):
        super().__init__()
        self.fuser = nn.Linear(concat_embeds_dim, embed_dim)

    def forward(self, embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
        concat_embeds = torch.cat([embs1, embs2], dim=1)  # [B, 3072]
        fused = self.fuser(concat_embeds) # [B, embed_dim]
        return fused.unsqueeze(1)  # [B, 1, embed_dim]


# =============================================================================
# DecoderBlock:
# =============================================================================

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(512, elementwise_affine=False)
        self.ln_2 = nn.LayerNorm(512, elementwise_affine=False)
        self.ln_3 = nn.LayerNorm(512, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.1,
            bias=False,
            batch_first=True,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.1,
            bias=False,
            batch_first=True,
        )

        self.net = nn.Sequential(
            nn.Linear(512, 2048, bias=False),
            nn.GELU(),
            nn.Linear(2048, 512, bias=False),
        )
        self.dropout_layer = nn.Dropout(0.1)

        mask = torch.triu(torch.full((15, 15), float('-inf')), diagonal=1)
        self.register_buffer("future_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, images_embeddings: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        attn_mask = self.future_mask[:T, :T].to(x.device)
        x = x + self.attn(
            query=self.ln_1(x),
            key=self.ln_1(x),
            value=self.ln_1(x),
            attn_mask=attn_mask,
            need_weights=False
        )[0]

        x = x + self.cross_attn(
            query=self.ln_2(x),
            key=images_embeddings,
            value=images_embeddings,
            need_weights=False
        )[0]

        x = x + self.dropout_layer(self.net(self.ln_3(x)))
        return x


# =============================================================================
# TransformationDecoder
# =============================================================================

class TransformationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_embd = 512
        self.n_head = 8
        self.n_layer = 3
        self.max_seq_len = 15
        self.vocab_size = 13
        self.dropout = 0.1
        self.bias = False
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.n_embd)
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd, elementwise_affine=self.bias)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

    def forward(self, images_embeddings: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x, images_embeddings)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(
        self,
        images_embeddings: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.Tensor:
        B = images_embeddings.shape[0]
        device = images_embeddings.device
        total_len = 1 + max_new_tokens
        assert total_len <= self.max_seq_len

        idx = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self(images_embeddings, idx)
            next_logits = logits / temperature
            next_logits[:, self.pad_token_id] = -float('inf')
            next_logits[:, self.bos_token_id] = -float('inf')

            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                idx_next = torch.argmax(next_logits, dim=-1)

            newly_finished = (idx_next == self.eos_token_id)
            finished = finished | newly_finished
            idx = torch.cat([idx, idx_next.unsqueeze(-1)], dim=1)

            if finished.all():
                break

        if idx.size(1) < total_len:
            pad = torch.full(
                (B, total_len - idx.size(1)),
                self.pad_token_id,
                device=device,
                dtype=torch.long
            )
            idx = torch.cat([idx, pad], dim=1)
        else:
            idx = idx[:, :total_len]

        for i in range(B):
            eos_pos = (idx[i] == self.eos_token_id).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0:
                idx[i, eos_pos[0].item() + 1:] = self.pad_token_id

        return idx