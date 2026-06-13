"""
Hard-case tuning loop for ImageTransformPlagiarismPredictor.

Focus of this loop: tighten the model on "tricky" negatives — pairs that LOOK
similar but are NOT plagiarism. Curated hard-negative pairs come from a JSON
file (same format consumed by run_tune_plagiarism.py).

Per input sample ``(img1, img2, img1_aug, img2_aug, seq1, seq2)`` we construct
10 * B pair instances:
    8 * B negatives (cross between the two curated-different images +/- aug)
    2 * B positives (augmented self-pairs: (img1, img1_aug), (img2, img2_aug))

Losses (same composition as pretraining, with tunable weights):
    total = w_seq * seq_CE + w_bce * BCE + w_nce * InfoNCE
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plagiarism.effnet_plagiarism import (
    BinaryMatchMetrics,
    info_nce_loss,
    pairwise_bce_loss,
)


# ======================
# Helpers
# ======================

def create_empty_sequence(batch_size, max_seq_len, bos_id, eos_id, pad_id):
    """[BOS, EOS, PAD, PAD, ...] — the sequence we assign to negative pairs."""
    seq = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
    seq[:, 0] = bos_id
    if max_seq_len > 1:
        seq[:, 1] = eos_id
    return seq


def get_optimizer(net, config):
    name = config["optimizer"]["name"]
    params = filter(lambda p: p.requires_grad, net.parameters())
    if name == "Adam":
        return torch.optim.Adam(
            params,
            lr=config["optimizer"]["lr"],
            betas=tuple(config["optimizer"]["betas"]),
            weight_decay=config["optimizer"]["weight_decay"],
        )
    if name == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=config["optimizer"]["lr"],
            betas=tuple(config["optimizer"]["betas"]),
            weight_decay=config["optimizer"]["weight_decay"],
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def get_scheduler(opt, config):
    return torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=list(config["scheduler"]["milestones"]),
        gamma=float(config["scheduler"]["gamma"]),
    )


def save_checkpoint(model, optimizer, scheduler, epoch, config):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    ckpt_dir = config["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(ckpt, path)
    print(f"[ckpt] saved epoch={epoch} -> {path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path) -> int:
    if not os.path.exists(checkpoint_path):
        print("[ckpt] no checkpoint found, starting from scratch")
        return 0
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    scheduler.load_state_dict(ck["scheduler_state_dict"])
    print(f"[ckpt] loaded epoch={ck['epoch']} from {checkpoint_path}")
    return int(ck["epoch"])


# ======================
# Main tuning loop
# ======================

def train_model(
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: DictConfig,
    tokenizer: Any,
) -> None:
    optimizer = get_optimizer(model, config)
    lr_scheduler = get_scheduler(optimizer, config)

    device = torch.device(config["training"]["device"])
    model.to(device)

    pad_token_id = int(config.model.decoder.pad_token_id)
    bos_token_id = int(config.model.decoder.bos_token_id)
    eos_token_id = int(config.model.decoder.eos_token_id)
    max_seq_len = int(config.model.decoder.max_seq_len)

    lw = config["training"].get("loss_weights", {})
    w_seq = float(lw.get("seq", 1.0))
    w_bce = float(lw.get("bce", 1.0))
    w_nce = float(lw.get("info_nce", 0.2))

    info_nce_temperature = float(config["training"].get("info_nce_temperature", 0.07))
    binary_threshold = float(config["training"].get("binary_threshold", 0.5))

    bce_pos_weight_scalar = config["training"].get("bce_pos_weight", None)
    bce_pos_weight = (
        torch.tensor([float(bce_pos_weight_scalar)], device=device)
        if bce_pos_weight_scalar is not None
        else None
    )

    num_epochs = int(config["training"]["num_epochs"])
    checkpoint_interval = int(config["training"]["checkpoint_interval"])
    checkpoint_dir = config["training"]["checkpoint_dir"]
    log_dir = config["data"]["tensorboard_logdir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0
    if config["training"]["resume"]:
        ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if ckpts:
            latest = max(
                [os.path.join(checkpoint_dir, f) for f in ckpts],
                key=os.path.getctime,
            )
            start_epoch = load_checkpoint(model, optimizer, lr_scheduler, latest)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_seq_len=max_seq_len,
            w_seq=w_seq,
            w_bce=w_bce,
            w_nce=w_nce,
            info_nce_temperature=info_nce_temperature,
            bce_pos_weight=bce_pos_weight,
            binary_threshold=binary_threshold,
            mode="train",
            desc=f"Train Epoch {epoch + 1}",
        )

        lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            val_stats = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_seq_len=max_seq_len,
                w_seq=w_seq,
                w_bce=w_bce,
                w_nce=w_nce,
                info_nce_temperature=info_nce_temperature,
                bce_pos_weight=bce_pos_weight,
                binary_threshold=binary_threshold,
                mode="val",
                desc=f"Val   Epoch {epoch + 1}",
            )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print(
            f"  Train Total {train_stats['total']:.4f} | Seq {train_stats['seq']:.4f} "
            f"| BCE {train_stats['bce']:.4f} | NCE {train_stats['nce']:.4f}"
        )
        print(
            f"  Train Prec {train_stats['precision']:.4f} | Recall {train_stats['recall']:.4f} "
            f"| F1 {train_stats['f1']:.4f} | FPR {train_stats['fpr']:.4f}"
        )
        print(
            f"  Val   Total {val_stats['total']:.4f} | Seq {val_stats['seq']:.4f} "
            f"| BCE {val_stats['bce']:.4f} | NCE {val_stats['nce']:.4f}"
        )
        print(
            f"  Val   Prec {val_stats['precision']:.4f} | Recall {val_stats['recall']:.4f} "
            f"| F1 {val_stats['f1']:.4f} | FPR {val_stats['fpr']:.4f}"
        )
        print(f"  LR {current_lr:.2e}")

        for split, s in (("Train", train_stats), ("Val", val_stats)):
            writer.add_scalar(f"Loss/{split}/Total", s["total"], epoch)
            writer.add_scalar(f"Loss/{split}/Seq", s["seq"], epoch)
            writer.add_scalar(f"Loss/{split}/BCE", s["bce"], epoch)
            writer.add_scalar(f"Loss/{split}/InfoNCE", s["nce"], epoch)
            writer.add_scalar(f"Binary/{split}/Precision", s["precision"], epoch)
            writer.add_scalar(f"Binary/{split}/Recall", s["recall"], epoch)
            writer.add_scalar(f"Binary/{split}/F1", s["f1"], epoch)
            writer.add_scalar(f"Binary/{split}/FPR", s["fpr"], epoch)

        writer.add_scalar("LearningRate", current_lr, epoch)

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, lr_scheduler, epoch + 1, config)

    writer.close()
    print("Tuning completed.")


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    pad_token_id,
    bos_token_id,
    eos_token_id,
    max_seq_len,
    w_seq,
    w_bce,
    w_nce,
    info_nce_temperature,
    bce_pos_weight: Optional[torch.Tensor],
    binary_threshold,
    mode,
    desc,
) -> Dict[str, float]:
    is_train = mode == "train"
    total_sum = seq_sum = bce_sum = nce_sum = 0.0
    total_pairs = 0
    metrics = BinaryMatchMetrics(threshold=binary_threshold)

    for batch in tqdm(loader, desc=desc):
        img1, img2, img1_aug, img2_aug, seq1, seq2 = [x.to(device, non_blocking=True) for x in batch]
        B = img1.size(0)

        if is_train:
            optimizer.zero_grad()

        # Per-image global features (one backbone pass per variant)
        # Returns [B, 1, feature_dim] each.
        f1 = model.image_pair_encoder.image_encoder(img1)
        f2 = model.image_pair_encoder.image_encoder(img2)
        f1a = model.image_pair_encoder.image_encoder(img1_aug)
        f2a = model.image_pair_encoder.image_encoder(img2_aug)

        # -------- Build 10B symmetric pair set --------
        # 8B negatives + 2B positives
        embs_a = torch.cat([
            f1,  f2,   # (1,2), (2,1)   negatives
            f1,  f2a,  # (1,2a), (2a,1)
            f2,  f1a,  # (2,1a), (1a,2)
            f2a, f1a,  # (2a,1a), (1a,2a)
            f1,  f2,   # (1,1a), (2,2a) positives
        ], dim=0)  # [10B, 1, feature_dim]

        embs_b = torch.cat([
            f2,  f1,
            f2a, f1,
            f1a, f2,
            f1a, f2a,
            f1a, f2a,
        ], dim=0)

        target_classes = torch.cat([
            torch.zeros(8 * B, dtype=torch.long, device=device),
            torch.ones(2 * B, dtype=torch.long, device=device),
        ], dim=0)

        # Decoder input: negatives get [BOS, EOS, PAD...], positives get the real seq.
        empty_seq = create_empty_sequence(
            8 * B, max_seq_len, bos_token_id, eos_token_id, pad_token_id
        ).to(device)
        pos_seq = torch.cat([seq1, seq2], dim=0)  # 2B
        full_idx = torch.cat([empty_seq, pos_seq], dim=0)

        # -------- Forward --------
        seq_logits, seq_loss, match_logits, _fused = model(
            embs_a,
            embs_b,
            full_idx,
            use_precomputed_embeddings=True,
            return_match_logit=True,
        )

        # -------- BCE --------
        bce = pairwise_bce_loss(
            match_logits=match_logits,
            labels=target_classes,
            pos_weight=bce_pos_weight,
        )

        # -------- InfoNCE on RAW per-image features --------
        # Anchors: f1, f2 ; positives: f1a, f2a.
        if w_nce > 0.0:
            z_orig = model.project_features(torch.cat([f1, f2], dim=0))  # [2B, D_proj]
            z_aug = model.project_features(torch.cat([f1a, f2a], dim=0))
            nce = info_nce_loss(z_orig, z_aug, temperature=info_nce_temperature)
        else:
            nce = torch.zeros((), device=device)

        total_loss = w_seq * seq_loss + w_bce * bce + w_nce * nce

        if is_train:
            total_loss.backward()
            optimizer.step()

        metrics.update(match_logits.detach(), target_classes.detach())

        n_pairs = 10 * B
        total_sum += float(total_loss.item()) * n_pairs
        seq_sum += float(seq_loss.item()) * n_pairs
        bce_sum += float(bce.item()) * n_pairs
        nce_sum += float(nce.item()) * n_pairs
        total_pairs += n_pairs

    return {
        "total": total_sum / max(total_pairs, 1),
        "seq": seq_sum / max(total_pairs, 1),
        "bce": bce_sum / max(total_pairs, 1),
        "nce": nce_sum / max(total_pairs, 1),
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "fpr": metrics.fpr,
    }
