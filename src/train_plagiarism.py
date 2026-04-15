"""
Pretraining loop for ImageTransformPlagiarismPredictor.

Based on ``train_complex.py`` but adapted for the new plagiarism-aware model:
  * BxB random cross-pairing inside each batch (single image loader with
    augmentations — same as before).
  * Adds an explicit binary cross-entropy on the match-head.
  * Adds a symmetric InfoNCE on projected per-image global features
    (positive = (orig_i, aug_i), negatives = all other sample-aug
    combinations in the batch).
  * Keeps the autoregressive CE on the transform sequence.

Expected data loader: yields (orig_batch, aug_batch, idx_batch).
"""

import os
from typing import Any, Dict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .model.effnet_plagiarism import (
    BinaryMatchMetrics,
    SeqBinaryMetrics,
    SeqTokenAccuracy,
    info_nce_loss,
    pairwise_bce_loss,
)


# ======================
# Utility Functions
# ======================

def create_negative_idx(
    batch_size: int,
    max_seq_len: int,
    start_token_id: int,
    end_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Creates idx sequences for negative pairs: [START, END, PAD, PAD, ...]."""
    idx = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    idx[:, 0] = start_token_id
    if max_seq_len > 1:
        idx[:, 1] = end_token_id
    return idx


def get_optimizer(net: torch.nn.Module, config: DictConfig) -> torch.optim.Optimizer:
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


def get_scheduler(opt: torch.optim.Optimizer, config: DictConfig):
    return torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=list(config["scheduler"]["milestones"]),
        gamma=float(config["scheduler"]["gamma"]),
    )


def save_checkpoint(model, optimizer, scheduler, epoch, config, augmentation_scheduler=None):
    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if augmentation_scheduler is not None:
        checkpoint["augmentation_scheduler_state_dict"] = augmentation_scheduler.state_dict()

    ckpt_dir = config["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, path)
    print(f"[ckpt] saved epoch={epoch} -> {path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, augmentation_scheduler=None) -> int:
    if not os.path.exists(checkpoint_path):
        print("[ckpt] no checkpoint found, starting from scratch")
        return 0
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    scheduler.load_state_dict(ck["scheduler_state_dict"])
    if augmentation_scheduler is not None and "augmentation_scheduler_state_dict" in ck:
        augmentation_scheduler.load_state_dict(ck["augmentation_scheduler_state_dict"])
    print(f"[ckpt] loaded epoch={ck['epoch']} from {checkpoint_path}")
    return int(ck["epoch"])


# ======================
# Main training loop
# ======================

def train_model(
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: DictConfig,
    augmentation_scheduler: Any,
) -> None:
    optimizer = get_optimizer(model, config)
    lr_scheduler = get_scheduler(optimizer, config)

    device = torch.device(config["training"]["device"])
    model.to(device)

    pad_token_id = int(config.model.decoder.pad_token_id)
    bos_token_id = int(config.model.decoder.bos_token_id)
    eos_token_id = int(config.model.decoder.eos_token_id)
    max_seq_len = int(config.model.decoder.max_seq_len)

    # Loss weights
    lw = config["training"].get("loss_weights", {})
    w_seq = float(lw.get("seq", 1.0))
    w_bce = float(lw.get("bce", 0.5))
    w_nce = float(lw.get("info_nce", 0.2))

    info_nce_temperature = float(config["training"].get("info_nce_temperature", 0.07))
    binary_threshold = float(config["training"].get("binary_threshold", 0.5))

    # Symmetric negatives: if True, also include mirror negative pairs (aug_j, orig_i)
    # for every forward negative (orig_i, aug_j), so the fuser+decoder learn an
    # order-invariant decision boundary. Positives stay forward-only because we do
    # NOT have ground-truth inverse transform sequences for (aug -> orig) direction.
    symmetric_negatives = bool(config["training"].get("symmetric_negatives", False))

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
            start_epoch = load_checkpoint(
                model, optimizer, lr_scheduler, latest, augmentation_scheduler
            )

    for epoch in range(start_epoch, num_epochs):
        if augmentation_scheduler is not None:
            augmentation_scheduler.step()
            current_aug_p = augmentation_scheduler.p
            print(f"[aug] epoch {epoch + 1}: p = {current_aug_p:.3f}")
        else:
            current_aug_p = 0.0

        # ---------------- Train ----------------
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
            symmetric_negatives=symmetric_negatives,
        )

        lr_scheduler.step()

        # ---------------- Val ------------------
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
                symmetric_negatives=symmetric_negatives,
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
            f"  Train TokAcc {train_stats['tok_acc']:.4f} | SeqPrec {train_stats['seq_precision']:.4f} "
            f"| SeqRec {train_stats['seq_recall']:.4f} | SeqF1 {train_stats['seq_f1']:.4f} "
            f"| SeqFPR {train_stats['seq_fpr']:.4f}"
        )
        print(
            f"  Val   Total {val_stats['total']:.4f} | Seq {val_stats['seq']:.4f} "
            f"| BCE {val_stats['bce']:.4f} | NCE {val_stats['nce']:.4f}"
        )
        print(
            f"  Val   Prec {val_stats['precision']:.4f} | Recall {val_stats['recall']:.4f} "
            f"| F1 {val_stats['f1']:.4f} | FPR {val_stats['fpr']:.4f}"
        )
        print(
            f"  Val   TokAcc {val_stats['tok_acc']:.4f} | SeqPrec {val_stats['seq_precision']:.4f} "
            f"| SeqRec {val_stats['seq_recall']:.4f} | SeqF1 {val_stats['seq_f1']:.4f} "
            f"| SeqFPR {val_stats['seq_fpr']:.4f}"
        )
        print(f"  LR {current_lr:.2e} | aug_p {current_aug_p:.3f}")

        for split, s in (("Train", train_stats), ("Val", val_stats)):
            writer.add_scalar(f"Loss/{split}/Total", s["total"], epoch)
            writer.add_scalar(f"Loss/{split}/Seq", s["seq"], epoch)
            writer.add_scalar(f"Loss/{split}/BCE", s["bce"], epoch)
            writer.add_scalar(f"Loss/{split}/InfoNCE", s["nce"], epoch)
            writer.add_scalar(f"Binary/{split}/Precision", s["precision"], epoch)
            writer.add_scalar(f"Binary/{split}/Recall", s["recall"], epoch)
            writer.add_scalar(f"Binary/{split}/F1", s["f1"], epoch)
            writer.add_scalar(f"Binary/{split}/FPR", s["fpr"], epoch)
            writer.add_scalar(f"Seq/{split}/TokenAccuracy", s["tok_acc"], epoch)
            writer.add_scalar(f"Seq/{split}/Precision", s["seq_precision"], epoch)
            writer.add_scalar(f"Seq/{split}/Recall", s["seq_recall"], epoch)
            writer.add_scalar(f"Seq/{split}/F1", s["seq_f1"], epoch)
            writer.add_scalar(f"Seq/{split}/FPR", s["seq_fpr"], epoch)

        writer.add_scalar("LearningRate", current_lr, epoch)
        writer.add_scalar("Augmentation/p", current_aug_p, epoch)

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, lr_scheduler, epoch + 1, config, augmentation_scheduler
            )

    writer.close()
    print("Pretraining completed.")


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
    bce_pos_weight,
    binary_threshold,
    mode,
    desc,
    symmetric_negatives: bool = False,
) -> Dict[str, float]:
    is_train = mode == "train"
    total_sum = seq_sum = bce_sum = nce_sum = 0.0
    total_pairs = 0
    metrics = BinaryMatchMetrics(threshold=binary_threshold)
    seq_acc = SeqTokenAccuracy(pad_token_id=pad_token_id)
    seq_bin = SeqBinaryMetrics(eos_token_id=eos_token_id)

    for orig_batch, aug_batch, idx_batch in tqdm(loader, desc=desc):
        B = orig_batch.size(0)
        orig_batch = orig_batch.to(device, non_blocking=True)
        aug_batch = aug_batch.to(device, non_blocking=True)
        idx_batch = idx_batch.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        # Per-image features (single backbone pass per batch)
        orig_feats, aug_feats = model.extract_image_embeddings(orig_batch, aug_batch)
        # orig_feats, aug_feats: [B, 1, feature_dim]
        _, L, D_enc = orig_feats.shape

        # -------- BxB cross pairing --------
        orig_all = orig_feats.unsqueeze(1).expand(B, B, L, D_enc).reshape(B * B, L, D_enc)
        aug_all = aug_feats.unsqueeze(0).expand(B, B, L, D_enc).reshape(B * B, L, D_enc)

        pos_idx = idx_batch  # [B, T] — real transform seq
        neg_idx = create_negative_idx(
            batch_size=B * (B - 1),
            max_seq_len=max_seq_len,
            start_token_id=bos_token_id,
            end_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        ).to(device)

        full_idx = torch.zeros(B * B, max_seq_len, dtype=torch.long, device=device)
        diag = torch.arange(B, device=device) * (B + 1)
        full_idx[diag] = pos_idx
        off = torch.ones(B * B, dtype=torch.bool, device=device)
        off[diag] = False
        full_idx[off] = neg_idx

        target_classes = torch.zeros(B * B, dtype=torch.long, device=device)
        target_classes[diag] = 1

        # -------- Symmetric negatives: mirror (aug_j, orig_i) for i != j --------
        if symmetric_negatives:
            sym_orig_base = orig_all[off]   # [B*(B-1), L, D_enc]
            sym_aug_base = aug_all[off]     # [B*(B-1), L, D_enc]

            orig_all = torch.cat([orig_all, sym_aug_base], dim=0)
            aug_all = torch.cat([aug_all, sym_orig_base], dim=0)
            full_idx = torch.cat([full_idx, neg_idx], dim=0)
            target_classes = torch.cat(
                [target_classes, torch.zeros(B * (B - 1), dtype=torch.long, device=device)],
                dim=0,
            )

        # -------- Forward --------
        seq_logits, seq_loss, match_logits, _fused = model(
            orig_all,
            aug_all,
            full_idx,
            use_precomputed_embeddings=True,
            return_match_logit=True,
        )

        # -------- BCE on match head --------
        bce = pairwise_bce_loss(
            match_logits=match_logits,
            labels=target_classes,
            pos_weight=bce_pos_weight,
        )

        # -------- InfoNCE on RAW per-image features --------
        # Positive pairs: (orig_i, aug_i). B anchors -> B positives, others are negatives.
        if w_nce > 0.0:
            z_orig = model.project_features(orig_feats)  # [B, D_proj]
            z_aug = model.project_features(aug_feats)
            nce = info_nce_loss(z_orig, z_aug, temperature=info_nce_temperature)
        else:
            nce = torch.zeros((), device=device)

        total_loss = w_seq * seq_loss + w_bce * bce + w_nce * nce

        if is_train:
            total_loss.backward()
            optimizer.step()

        # Metrics
        metrics.update(match_logits.detach(), target_classes.detach())
        seq_acc.update(seq_logits.detach(), full_idx)
        seq_bin.update(seq_logits.detach(), target_classes)

        n_pairs = orig_all.size(0)
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
        "tok_acc": seq_acc.accuracy,
        "seq_precision": seq_bin.precision,
        "seq_recall": seq_bin.recall,
        "seq_f1": seq_bin.f1,
        "seq_fpr": seq_bin.fpr,
    }
