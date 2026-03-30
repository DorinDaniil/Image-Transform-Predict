import os
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ======================
# Utility Functions
# ======================

def create_negative_idx(batch_size, max_seq_len, start_token_id, end_token_id, pad_token_id):
    """
    Creates `idx` sequences for negative pairs: [START, END, PAD, PAD, ...].
    Returns a tensor of shape [batch_size, max_seq_len].
    """
    idx = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    idx[:, 0] = start_token_id
    if max_seq_len > 1:
        idx[:, 1] = end_token_id
    return idx


# ======================
# Configuration and Checkpoint Management
# ======================

def get_optimizer(net, config):
    """Initialize optimizer based on config."""
    optimizer_name = config['optimizer']['name']
    if optimizer_name == 'Adam':
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
    elif optimizer_name == 'AdamW':
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return opt


def get_scheduler(opt, config):
    """Initialize learning rate scheduler."""
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=config['scheduler']['milestones'],
        gamma=config['scheduler']['gamma']
    )
    return sched


def save_checkpoint(model, optimizer, scheduler, epoch, config, augmentation_scheduler=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    if augmentation_scheduler is not None:
        checkpoint['augmentation_scheduler_state_dict'] = augmentation_scheduler.state_dict()

    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, augmentation_scheduler=None):
    """Load model checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        if augmentation_scheduler is not None and 'augmentation_scheduler_state_dict' in checkpoint:
            augmentation_scheduler.load_state_dict(checkpoint['augmentation_scheduler_state_dict'])
        print(f'Checkpoint loaded from epoch {epoch}')
        return epoch
    else:
        print('No checkpoint found. Starting from scratch.')
        return 0


# ==============================================================================
# Epoch-Aggregated Metrics (скопированы из tuning.py)
# ==============================================================================


class EpochSequenceAccuracy:
    """Exact match accuracy: % of fully correct sequences over entire epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_seqs = 0
        self.total_seqs = 0

    def update(self, pred_logits: torch.Tensor, target_ids: torch.Tensor, pad_token_id: int):
        pred_ids = pred_logits.argmax(dim=-1)
        mask = target_ids != pad_token_id
        correct = (pred_ids == target_ids) | (~mask)
        seq_correct = correct.all(dim=1)
        self.correct_seqs += seq_correct.sum().item()
        self.total_seqs += target_ids.size(0)

    @property
    def value(self) -> float:
        return self.correct_seqs / self.total_seqs if self.total_seqs > 0 else 0.0


class EpochTokenAccuracy:
    """Token accuracy over non-padding positions (epoch-level)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_tokens = 0
        self.total_tokens = 0

    def update(self, pred_logits: torch.Tensor, target_ids: torch.Tensor, pad_token_id: int):
        pred_ids = pred_logits.argmax(dim=-1)
        mask = target_ids != pad_token_id
        correct = (pred_ids == target_ids) & mask
        self.correct_tokens += correct.sum().item()
        self.total_tokens += mask.sum().item()

    @property
    def value(self) -> float:
        return self.correct_tokens / self.total_tokens if self.total_tokens > 0 else 0.0


class EpochPrecisionRecall:
    """Binary classification metrics (0 = negative pair, 1 = positive pair)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.detach().cpu().long()
        targets = targets.detach().cpu().long()
        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r > 0 else 0.0


# ==============================================================================
# Training loop with tuning-style metrics but original train pairing
# ==============================================================================


def train_model(
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: Any,
    augmentation_scheduler: Any,
) -> None:

    optimizer = get_optimizer(model, config)
    lr_scheduler = get_scheduler(optimizer, config)

    num_epochs = config["training"]["num_epochs"]
    device = torch.device(config["training"]["device"])
    checkpoint_interval = config["training"]["checkpoint_interval"]
    checkpoint_dir = config["training"]["checkpoint_dir"]
    log_dir = config["data"]["tensorboard_logdir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model.to(device)

    pad_token_id = config.model.decoder.pad_token_id
    bos_token_id = config.model.decoder.bos_token_id
    eos_token_id = config.model.decoder.eos_token_id
    max_seq_len = config.model.decoder.max_seq_len

    # Resume
    start_epoch = 0
    if config["training"]["resume"]:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if checkpoints:
            latest_checkpoint = max(
                [os.path.join(checkpoint_dir, f) for f in checkpoints],
                key=os.path.getctime,
            )
            start_epoch = load_checkpoint(model, optimizer, lr_scheduler, latest_checkpoint, augmentation_scheduler)

    for epoch in range(start_epoch, num_epochs):
        augmentation_scheduler.step()
        current_aug_p = augmentation_scheduler.p
        print(f"[Augmentation] Epoch {epoch + 1}: p = {current_aug_p:.3f}")

        # === Train ===
        model.train()
        seq_acc_train = EpochSequenceAccuracy()
        tok_acc_train = EpochTokenAccuracy()
        pr_train = EpochPrecisionRecall()
        train_loss = 0.0
        total_pairs = 0

        for orig_batch, aug_batch, idx_batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            B = orig_batch.size(0)
            orig_batch = orig_batch.to(device)
            aug_batch = aug_batch.to(device)
            idx_batch = idx_batch.to(device)

            optimizer.zero_grad()

            orig_batch_features, aug_batch_features = model.extract_image_embeddings(orig_batch, aug_batch)
            B, L, D = orig_batch_features.shape

            orig_all_features = orig_batch_features.unsqueeze(1).expand(B, B, L, D)
            aug_all_features = aug_batch_features.unsqueeze(0).expand(B, B, L, D)

            orig_all_features = orig_all_features.reshape(B * B, L, D)
            aug_all_features = aug_all_features.reshape(B * B, L, D)

            pos_idx = idx_batch
            neg_idx = create_negative_idx(
                batch_size=B * (B - 1),
                max_seq_len=max_seq_len,
                start_token_id=bos_token_id,
                end_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            ).to(device)

            full_idx = torch.zeros(B * B, max_seq_len, dtype=torch.long, device=device)
            diag_indices = torch.arange(B, device=device) * (B + 1)
            full_idx[diag_indices] = pos_idx
            off_diag_mask = torch.ones(B * B, dtype=torch.bool, device=device)
            off_diag_mask[diag_indices] = False
            full_idx[off_diag_mask] = neg_idx

            full_targets = torch.full_like(full_idx, pad_token_id)
            full_targets[:, :-1] = full_idx[:, 1:]

            # Бинарные таргеты: 1 для диагональных (позитив), 0 для остальных (негатив)
            target_classes = torch.zeros(B * B, dtype=torch.long, device=device)
            target_classes[diag_indices] = 1

            logits, loss = model(
                orig_all_features,
                aug_all_features,
                full_idx,
                use_precomputed_embeddings=True,
            )

            # seq/token accuracy
            seq_acc_train.update(logits, full_targets, pad_token_id)
            tok_acc_train.update(logits, full_targets, pad_token_id)

            # binary cls по первому токену, как в tuning.py
            first_pred_token = logits[:, 0, :].argmax(dim=-1)
            preds_cls = (first_pred_token != eos_token_id).long()
            pr_train.update(preds_cls, target_classes)

            loss.backward()
            optimizer.step()

            n_pairs = B * B
            train_loss += loss.item() * n_pairs
            total_pairs += n_pairs

        lr_scheduler.step()

        avg_train_loss = train_loss / total_pairs
        avg_train_seq_acc = seq_acc_train.value
        avg_train_token_acc = tok_acc_train.value
        train_precision = pr_train.precision
        train_recall = pr_train.recall
        train_f1 = pr_train.f1

        # === Val ===
        model.eval()
        seq_acc_val = EpochSequenceAccuracy()
        tok_acc_val = EpochTokenAccuracy()
        pr_val = EpochPrecisionRecall()
        val_loss = 0.0
        val_total_pairs = 0

        with torch.no_grad():
            for orig_batch, aug_batch, idx_batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                B = orig_batch.size(0)
                orig_batch = orig_batch.to(device)
                aug_batch = aug_batch.to(device)
                idx_batch = idx_batch.to(device)

                orig_batch_features, aug_batch_features = model.extract_image_embeddings(orig_batch, aug_batch)
                B, L, D = orig_batch_features.shape

                orig_all_features = orig_batch_features.unsqueeze(1).expand(B, B, L, D)
                aug_all_features = aug_batch_features.unsqueeze(0).expand(B, B, L, D)

                orig_all_features = orig_all_features.reshape(B * B, L, D)
                aug_all_features = aug_all_features.reshape(B * B, L, D)

                pos_idx = idx_batch
                neg_idx = create_negative_idx(
                    batch_size=B * (B - 1),
                    max_seq_len=max_seq_len,
                    start_token_id=bos_token_id,
                    end_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                ).to(device)

                full_idx = torch.zeros(B * B, max_seq_len, dtype=torch.long, device=device)
                diag_indices = torch.arange(B, device=device) * (B + 1)
                full_idx[diag_indices] = pos_idx
                off_diag_mask = torch.ones(B * B, dtype=torch.bool, device=device)
                off_diag_mask[diag_indices] = False
                full_idx[off_diag_mask] = neg_idx

                full_targets = torch.full_like(full_idx, pad_token_id)
                full_targets[:, :-1] = full_idx[:, 1:]

                target_classes = torch.zeros(B * B, dtype=torch.long, device=device)
                target_classes[diag_indices] = 1

                logits, loss = model(
                    orig_all_features,
                    aug_all_features,
                    full_idx,
                    use_precomputed_embeddings=True,
                )

                seq_acc_val.update(logits, full_targets, pad_token_id)
                tok_acc_val.update(logits, full_targets, pad_token_id)

                first_pred_token = logits[:, 0, :].argmax(dim=-1)
                preds_cls = (first_pred_token != eos_token_id).long()
                pr_val.update(preds_cls, target_classes)

                n_pairs = B * B
                val_loss += loss.item() * n_pairs
                val_total_pairs += n_pairs

        avg_val_loss = val_loss / val_total_pairs
        avg_val_seq_acc = seq_acc_val.value
        avg_val_token_acc = tok_acc_val.value
        val_precision = pr_val.precision
        val_recall = pr_val.recall
        val_f1 = pr_val.f1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | SeqAcc: {avg_train_seq_acc:.4f} | TokAcc: {avg_train_token_acc:.4f}")
        print(f"  Train Prec: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | SeqAcc: {avg_val_seq_acc:.4f} | TokAcc: {avg_val_token_acc:.4f}")
        print(f"  Val   Prec: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.2e}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("SeqAcc/Train", avg_train_seq_acc, epoch)
        writer.add_scalar("TokAcc/Train", avg_train_token_acc, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("SeqAcc/Val", avg_val_seq_acc, epoch)
        writer.add_scalar("TokAcc/Val", avg_val_token_acc, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)
        writer.add_scalar("Recall/Train", train_recall, epoch)
        writer.add_scalar("F1/Train", train_f1, epoch)
        writer.add_scalar("Precision/Val", val_precision, epoch)
        writer.add_scalar("Recall/Val", val_recall, epoch)
        writer.add_scalar("F1/Val", val_f1, epoch)
        writer.add_scalar("Learning Rate", current_lr, epoch)
        writer.add_scalar("Augmentation/p", current_aug_p, epoch)

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, lr_scheduler, epoch + 1, config, augmentation_scheduler)

    writer.close()
    print("Training with metrics completed.")