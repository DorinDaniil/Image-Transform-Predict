import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


# ==============================================================================
# Epoch-Aggregated Metrics
# ==============================================================================

class EpochSequenceAccuracy:
    """Exact match accuracy: % of fully correct sequences over entire epoch."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_seqs = 0
        self.total_seqs = 0

    def update(self, pred_logits, target_ids, pad_token_id):
        pred_ids = pred_logits.argmax(dim=-1)
        mask = target_ids != pad_token_id
        correct = (pred_ids == target_ids) | (~mask)
        seq_correct = correct.all(dim=1)
        self.correct_seqs += seq_correct.sum().item()
        self.total_seqs += target_ids.size(0)

    @property
    def value(self):
        return self.correct_seqs / self.total_seqs if self.total_seqs > 0 else 0.0


class EpochTokenAccuracy:
    """Token accuracy over non-padding positions (epoch-level)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_tokens = 0
        self.total_tokens = 0

    def update(self, pred_logits, target_ids, pad_token_id):
        pred_ids = pred_logits.argmax(dim=-1)
        mask = target_ids != pad_token_id
        correct = (pred_ids == target_ids) & mask
        self.correct_tokens += correct.sum().item()
        self.total_tokens += mask.sum().item()

    @property
    def value(self):
        return self.correct_tokens / self.total_tokens if self.total_tokens > 0 else 0.0


class EpochPrecisionRecall:
    """Binary classification metrics (0 = negative pair, 1 = positive pair)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, preds, targets):
        preds = preds.detach().cpu().long()
        targets = targets.detach().cpu().long()
        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()

    @property
    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r > 0 else 0.0


# ==============================================================================
# Utils
# ==============================================================================

def create_empty_sequence(batch_size, max_seq_len, tokenizer):
    pad_id = getattr(tokenizer, 'pad_id', 0)
    bos_id = getattr(tokenizer, 'bos_id', 1)
    eos_id = getattr(tokenizer, 'eos_id', 2)

    seq = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
    seq[:, 0] = bos_id
    if max_seq_len > 1:
        seq[:, 1] = eos_id
    return seq

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
# Training Loop
# ==============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    config,
    tokenizer
):
    optimizer = get_optimizer(model, config)
    lr_scheduler = get_scheduler(optimizer, config)

    device = torch.device(config['training']['device'])
    model.to(device)

    pad_token_id = config.model.decoder.pad_token_id
    eos_token_id = config.model.decoder.eos_token_id
    max_seq_len = config.model.decoder.max_seq_len

    # Setup logging
    log_dir = config['data']['tensorboard_logdir']
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Resume from checkpoint
    start_epoch = 0
    if config['training']['resume']:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            latest = max(
                [os.path.join(checkpoint_dir, f) for f in checkpoints],
                key=os.path.getctime
            )
            start_epoch = load_checkpoint(model, optimizer, lr_scheduler, latest)

    # Training loop
    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()

        # Initialize epoch metrics
        seq_acc_train = EpochSequenceAccuracy()
        tok_acc_train = EpochTokenAccuracy()
        pr_train = EpochPrecisionRecall()
        train_loss = 0.0
        total_pairs = 0

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            img1, img2, img1_aug, img2_aug, seq1, seq2 = [x.to(device) for x in batch]
            B = img1.size(0)

            optimizer.zero_grad()

            # Extract features: (B, L, D)
            f1, f2 = model.extract_image_embeddings(img1, img2)
            f1a, f2a = model.extract_image_embeddings(img1_aug, img2_aug)

            # Stack for 5B pairs: (img1,img2), (img1,img2_aug), (img2,img1_aug), (img1,img1_aug), (img2,img2_aug)
            A_feats = torch.cat([f1, f1, f2, f1, f2], dim=0)      # (5B, L, D)
            B_feats = torch.cat([f2, f2a, f1a, f1a, f2a], dim=0)  # (5B, L, D)

            # del f1, f2, f1a, f2a
            # torch.cuda.empty_cache()

            # Build idx sequences
            empty_seq = create_empty_sequence(3 * B, max_seq_len, tokenizer).to(device)
            pos_seq = torch.cat([seq1, seq2], dim=0)               # (2B, T)
            full_idx = torch.cat([empty_seq, pos_seq], dim=0)      # (5B, T)

            # Binary targets: 0 for negatives (first 3B), 1 for positives (last 2B)
            target_classes = torch.tensor(
                [0] * (3 * B) + [1] * (2 * B),
                dtype=torch.long, device=device
            )

            # Sequence targets (shifted for autoregressive loss)
            full_targets_seq = torch.full_like(full_idx, pad_token_id)
            full_targets_seq[:, :-1] = full_idx[:, 1:]

            # Forward pass
            logits, loss = model(
                A_feats, B_feats, full_idx,
                use_precomputed_embeddings=True
            )

            # Update metrics
            seq_acc_train.update(logits, full_targets_seq, pad_token_id)
            tok_acc_train.update(logits, full_targets_seq, pad_token_id)

            # Binary classification: class 0 if first predicted token is eos_token_id
            first_pred_token = logits[:, 0, :].argmax(dim=-1)
            preds_cls = (first_pred_token != eos_token_id).long()
            pr_train.update(preds_cls, target_classes)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * (5 * B)
            total_pairs += 5 * B

        lr_scheduler.step()

        avg_train_loss = train_loss / total_pairs
        avg_train_seq_acc = seq_acc_train.value
        avg_train_token_acc = tok_acc_train.value
        train_precision = pr_train.precision
        train_recall = pr_train.recall
        train_f1 = pr_train.f1

        # Validation
        model.eval()
        seq_acc_val = EpochSequenceAccuracy()
        tok_acc_val = EpochTokenAccuracy()
        pr_val = EpochPrecisionRecall()
        val_loss = 0.0
        val_total_pairs = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                img1, img2, img1_aug, img2_aug, seq1, seq2 = [x.to(device) for x in batch]
                B = img1.size(0)

                f1, f2 = model.extract_image_embeddings(img1, img2)
                f1a, f2a = model.extract_image_embeddings(img1_aug, img2_aug)

                A_feats = torch.cat([f1, f1, f2, f1, f2], dim=0)
                B_feats = torch.cat([f2, f2a, f1a, f1a, f2a], dim=0)

                empty_seq = create_empty_sequence(3 * B, max_seq_len, tokenizer).to(device)
                pos_seq = torch.cat([seq1, seq2], dim=0)
                full_idx = torch.cat([empty_seq, pos_seq], dim=0)

                target_classes = torch.tensor(
                    [0] * (3 * B) + [1] * (2 * B),
                    dtype=torch.long, device=device
                )
                full_targets_seq = torch.full_like(full_idx, pad_token_id)
                full_targets_seq[:, :-1] = full_idx[:, 1:]

                logits, loss = model(
                    A_feats, B_feats, full_idx,
                    use_precomputed_embeddings=True
                )

                seq_acc_val.update(logits, full_targets_seq, pad_token_id)
                tok_acc_val.update(logits, full_targets_seq, pad_token_id)

                # Binary classification: class 0 if first predicted token is eos_token_id
                first_pred_token = logits[:, 0, :].argmax(dim=-1)
                preds_cls = (first_pred_token != eos_token_id).long()
                pr_val.update(preds_cls, target_classes)

                val_loss += loss.item() * (5 * B)
                val_total_pairs += 5 * B

        avg_val_loss = val_loss / val_total_pairs
        avg_val_seq_acc = seq_acc_val.value
        avg_val_token_acc = tok_acc_val.value
        val_precision = pr_val.precision
        val_recall = pr_val.recall
        val_f1 = pr_val.f1

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch+1}/{config['training']['num_epochs']}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | SeqAcc: {avg_train_seq_acc:.4f} | TokAcc: {avg_train_token_acc:.4f}")
        print(f"  Train Prec: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | SeqAcc: {avg_val_seq_acc:.4f} | TokAcc: {avg_val_token_acc:.4f}")
        print(f"  Val   Prec: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('SeqAcc/Train', avg_train_seq_acc, epoch)
        writer.add_scalar('SeqAcc/Val', avg_val_seq_acc, epoch)
        writer.add_scalar('TokAcc/Train', avg_train_token_acc, epoch)
        writer.add_scalar('TokAcc/Val', avg_val_token_acc, epoch)
        writer.add_scalar('Precision/Train', train_precision, epoch)
        writer.add_scalar('Recall/Train', train_recall, epoch)
        writer.add_scalar('F1/Train', train_f1, epoch)
        writer.add_scalar('Precision/Val', val_precision, epoch)
        writer.add_scalar('Recall/Val', val_recall, epoch)
        writer.add_scalar('F1/Val', val_f1, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)

        # Save checkpoint
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, lr_scheduler, epoch + 1, config)

    writer.close()
    print("Training completed.")