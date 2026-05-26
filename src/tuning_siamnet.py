import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import yaml


# ==============================================================================
# Epoch-Aggregated Metrics
# ==============================================================================

def init_metrics():
    return {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

def update_metrics(metrics_dict, true_labels, predicted_labels):
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(predicted_labels, torch.Tensor):
        predicted_labels = predicted_labels.cpu().numpy()

    tp = ((predicted_labels == 1) & (true_labels == 1)).sum()
    fp = ((predicted_labels == 1) & (true_labels == 0)).sum()
    tn = ((predicted_labels == 0) & (true_labels == 0)).sum()
    fn = ((predicted_labels == 0) & (true_labels == 1)).sum()

    metrics_dict['tp'] += int(tp)
    metrics_dict['fp'] += int(fp)
    metrics_dict['tn'] += int(tn)
    metrics_dict['fn'] += int(fn)
    return metrics_dict

def compute_metrics(metrics_dict):
    tp = metrics_dict['tp']
    fp = metrics_dict['fp']
    tn = metrics_dict['tn']
    fn = metrics_dict['fn']
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ==============================================================================
# Utils
# ==============================================================================

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_optimizer(net, config):
    optimizer_name = config['optimizer']['name']
    params = filter(lambda p: p.requires_grad, net.parameters())
    
    if optimizer_name == 'Adam':
        return torch.optim.Adam(
            params,
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(
            params,
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['scheduler']['milestones'],
        gamma=config['scheduler']['gamma']
    )

def save_checkpoint(model, optimizer, scheduler, epoch, config):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Checkpoint loaded from epoch {epoch}')
        return epoch
    else:
        print('No checkpoint found. Starting training from scratch.')
        return 0


# ==============================================================================
# Training Loop
# ==============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    config,
    resume=False
):
    device = torch.device(config['training']['device'])
    model.to(device)
    
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    if config['training'].get('contrastive_regularizer', False):
        contrastive_criterion = torch.nn.CosineEmbeddingLoss(
            margin=config['regularizer']['margin']
        )
        lambda_contrastive = config['regularizer']['lambda']
    else:
        contrastive_criterion = None
        lambda_contrastive = 0.0

    # TensorBoard
    log_dir = config['data']['tensorboard_logdir']
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0
    if resume:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            latest = max(
                [os.path.join(checkpoint_dir, f) for f in checkpoints],
                key=os.path.getctime
            )
            start_epoch = load_checkpoint(model, optimizer, scheduler, latest)

    num_epochs = config['training']['num_epochs']
    checkpoint_interval = config['training']['checkpoint_interval']

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss_total = 0.0
        train_contrastive_loss_total = 0.0
        train_bce_loss_total = 0.0
        train_total_pairs = 0
        train_metrics = init_metrics()

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            img1, img2, img1_aug, img2_aug, _, _ = [x for x in batch]
            B = img1.size(0)

            img1 = img1.to(device)
            img2 = img2.to(device)
            img1_aug = img1_aug.to(device)
            img2_aug = img2_aug.to(device)
            
            f1 = model.encode(img1)   # (B, D)
            f2 = model.encode(img2)   # (B, D)
            f1a = model.encode(img1_aug)  # (B, D)
            f2a = model.encode(img2_aug)  # (B, D)
            
            embs_A = torch.cat([f1, f1, f2, f1, f2], dim=0)      # (5B, D)
            embs_B = torch.cat([f2, f2a, f1a, f1a, f2a], dim=0)  # (5B, D)
            
            # Tags: first 3B - negative (0), last 2B - positive (1)
            target_classes = torch.tensor(
                [0] * (3 * B) + [1] * (2 * B),
                dtype=torch.float32,
                device=device
            )
            
            weights = torch.where(
                target_classes == 1,
                torch.full_like(target_classes, 0.3),
                torch.full_like(target_classes, 0.7)
            )

            optimizer.zero_grad()

            probabilities = model.predict_similarity(embs_A, embs_B, use_precomputed_embeddings=True).squeeze(-1)  # (5B,)
            
            # === Minimal label smoothing for BCE ===
            # eps = 0.05
            # smoothed_targets = torch.where(target_classes == 1.0, 1.0 - eps, eps)
            
            # bce_loss = F.binary_cross_entropy(probabilities, target_classes, weight=weights)
            bce_loss = F.binary_cross_entropy(probabilities, target_classes, weight=weights)

            total_loss = bce_loss

            contrastive_loss_value = 0.0
            if contrastive_criterion is not None:
                # Transform labels: {0,1} → {-1,1} for CosineEmbeddingLoss
                contrastive_labels = 2 * target_classes - 1
                
                contrastive_loss = contrastive_criterion(embs_A, embs_B, contrastive_labels)
                contrastive_loss_value = contrastive_loss.item()
                total_loss += lambda_contrastive * contrastive_loss

            total_loss.backward()
            optimizer.step()

            train_bce_loss_total += bce_loss.item() * (5 * B)
            if contrastive_criterion is not None:
                train_contrastive_loss_total += contrastive_loss_value * (5 * B)
            train_loss_total += total_loss.item() * (5 * B)
            train_total_pairs += 5 * B

            preds = (probabilities.detach() > 0.5).long()
            targets = target_classes.long()
            update_metrics(train_metrics, targets.cpu().numpy(), preds.cpu().numpy())

        scheduler.step()

        avg_train_loss = train_loss_total / train_total_pairs
        avg_train_bce = train_bce_loss_total / train_total_pairs
        avg_train_contrastive = train_contrastive_loss_total / train_total_pairs if contrastive_criterion else 0.0
        train_metrics_computed = compute_metrics(train_metrics)

        model.eval()
        val_loss_total = 0.0
        val_contrastive_loss_total = 0.0
        val_bce_loss_total = 0.0
        val_total_pairs = 0
        val_metrics = init_metrics()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                img1, img2, img1_aug, img2_aug, _, _ = [x for x in batch]
                B = img1.size(0)
    
                img1 = img1.to(device)
                img2 = img2.to(device)
                img1_aug = img1_aug.to(device)
                img2_aug = img2_aug.to(device)
                
                f1 = model.encode(img1)   # (B, D)
                f2 = model.encode(img2)   # (B, D)
                f1a = model.encode(img1_aug)  # (B, D)
                f2a = model.encode(img2_aug)  # (B, D)
                
                embs_A = torch.cat([f1, f1, f2, f1, f2], dim=0)      # (5B, D)
                embs_B = torch.cat([f2, f2a, f1a, f1a, f2a], dim=0)  # (5B, D)
                
                # Tags: first 3B - negative (0), last 2B - positive (1)
                target_classes = torch.tensor(
                    [0] * (3 * B) + [1] * (2 * B),
                    dtype=torch.float32,
                    device=device
                )
                
                # Weights for BCE: errors in class 0 (negative) are more expensive
                weights = torch.where(
                    target_classes == 1,
                    torch.full_like(target_classes, 0.3),
                    torch.full_like(target_classes, 0.7)
                )
    
                probabilities = model.predict_similarity(embs_A, embs_B, use_precomputed_embeddings=True).squeeze(-1)  # (5B,)

                # === Minimal label smoothing for BCE ===
                # eps = 0.05
                # smoothed_targets = torch.where(target_classes == 1.0, 1.0 - eps, eps)
                
                # bce_loss = F.binary_cross_entropy(probabilities, smoothed_targets, weight=weights)
                bce_loss = F.binary_cross_entropy(probabilities, target_classes, weight=weights)

                total_loss = bce_loss
    
                contrastive_loss_value = 0.0
                if contrastive_criterion is not None:
                    # Transform labels: {0,1} → {-1,1} for CosineEmbeddingLoss
                    contrastive_labels = 2 * target_classes - 1

                    contrastive_loss = contrastive_criterion(embs_A, embs_B, contrastive_labels)
                    contrastive_loss_value = contrastive_loss.item()
                    total_loss += lambda_contrastive * contrastive_loss

                val_bce_loss_total += bce_loss.item() * (5 * B)
                if contrastive_criterion is not None:
                    val_contrastive_loss_total += contrastive_loss_value * (5 * B)
                val_loss_total += total_loss.item() * (5 * B)
                val_total_pairs += 5 * B

                preds = (probabilities > 0.5).long()
                targets = target_classes.long()
                update_metrics(val_metrics, targets.cpu().numpy(), preds.cpu().numpy())

        avg_val_loss = val_loss_total / val_total_pairs
        avg_val_bce = val_bce_loss_total / val_total_pairs
        avg_val_contrastive = val_contrastive_loss_total / val_total_pairs if contrastive_criterion else 0.0
        val_metrics_computed = compute_metrics(val_metrics)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce:.4f}, Cont: {avg_train_contrastive:.4f})")
        print(f"  Train Acc: {train_metrics_computed['accuracy']:.4f} | Prec: {train_metrics_computed['precision']:.4f} | Rec: {train_metrics_computed['recall']:.4f} | F1: {train_metrics_computed['f1']:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} (BCE: {avg_val_bce:.4f}, Cont: {avg_val_contrastive:.4f})")
        print(f"  Val   Acc: {val_metrics_computed['accuracy']:.4f} | Prec: {val_metrics_computed['precision']:.4f} | Rec: {val_metrics_computed['recall']:.4f} | F1: {val_metrics_computed['f1']:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # TensorBoard
        writer.add_scalar('Loss/Train/Total', avg_train_loss, epoch)
        writer.add_scalar('Loss/Train/BCE', avg_train_bce, epoch)
        if contrastive_criterion:
            writer.add_scalar('Loss/Train/Contrastive', avg_train_contrastive, epoch)
        
        writer.add_scalar('Loss/Val/Total', avg_val_loss, epoch)
        writer.add_scalar('Loss/Val/BCE', avg_val_bce, epoch)
        if contrastive_criterion:
            writer.add_scalar('Loss/Val/Contrastive', avg_val_contrastive, epoch)
        
        writer.add_scalar('Metrics/Train/Accuracy', train_metrics_computed['accuracy'], epoch)
        writer.add_scalar('Metrics/Train/Precision', train_metrics_computed['precision'], epoch)
        writer.add_scalar('Metrics/Train/Recall', train_metrics_computed['recall'], epoch)
        writer.add_scalar('Metrics/Train/F1', train_metrics_computed['f1'], epoch)
        
        writer.add_scalar('Metrics/Val/Accuracy', val_metrics_computed['accuracy'], epoch)
        writer.add_scalar('Metrics/Val/Precision', val_metrics_computed['precision'], epoch)
        writer.add_scalar('Metrics/Val/Recall', val_metrics_computed['recall'], epoch)
        writer.add_scalar('Metrics/Val/F1', val_metrics_computed['f1'], epoch)
        
        writer.add_scalar('Learning Rate', current_lr, epoch)

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, config)

    writer.close()
    print("Training completed.")