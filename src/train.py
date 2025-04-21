import os
import argparse
import yaml
import time
import datetime
import json
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import BrainMRIDataset, get_transforms
from src.models import unet_scratch, unet_resnet34


def custom_collate(batch):
    """
    Custom collate function to ensure tensors can be batched properly.
    The dataset now handles shape normalization, so this is simpler.
    """
    # Return empty tensors if batch is empty
    if not batch:
        return torch.zeros(0), torch.zeros(0)
        
    images = []
    masks = []
    
    for item in batch:
        # Ensure each item is a tuple of image and mask
        if len(item) >= 2:
            # Clone the tensors to make them resizable
            image = item[0].clone().detach()
            mask = item[1].clone().detach()
            
            images.append(image)
            masks.append(mask)
    
    # Stack images and masks into batches if we have any valid samples
    if images and masks:
        try:
            images = torch.stack(images, 0)
            masks = torch.stack(masks, 0)
            return images, masks
        except RuntimeError as e:
            print(f"Error stacking tensors: {e}")
            # If still can't stack, just use the first item to create a batch of 1
            # The model will only see one sample but it will have the right dimensions
            return images[0].unsqueeze(0), masks[0].unsqueeze(0)
    else:
        print("Warning: Empty batch or inconsistent tensors, skipping...")
        # Return empty tensors
        return torch.zeros(0), torch.zeros(0)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dice_loss(pred, target, smooth=1.0):
    """Dice loss for segmentation."""
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1.0 - dice


def dice_coef(pred, target, threshold=0.5, smooth=1.0):
    """Dice coefficient for evaluation."""
    # Apply sigmoid and threshold
    pred = torch.sigmoid(pred) > threshold
    pred = pred.float()
    
    # Debug info on prediction and target
    all_zeros_pred = (pred.sum() == 0)
    all_zeros_target = (target.sum() == 0)
    
    if all_zeros_pred and all_zeros_target:
        # If both are empty, dice should be 1
        return 1.0
    
    # Move to 1D tensors
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    # Calculate intersection and dice
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def get_model(cfg):
    """Get model based on configuration."""
    model_cfg = cfg['model']
    model_type = model_cfg['type']
    
    if model_type == 'unet_scratch':
        model = unet_scratch.build(
            in_channels=model_cfg.get('in_channels', 1),
            n_classes=model_cfg.get('n_classes', 1),
            bilinear=model_cfg.get('bilinear', True)
        )
        freeze_epochs = 0
    
    elif model_type == 'unet_resnet34':
        model, freeze_epochs = unet_resnet34.build(
            in_channels=model_cfg.get('in_channels', 3),
            n_classes=model_cfg.get('n_classes', 1),
            pretrained=model_cfg.get('pretrained', True),
            freeze_epochs=model_cfg.get('freeze_epochs', 0)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, freeze_epochs


def get_optimizer(cfg, model_params):
    """Get optimizer based on configuration."""
    optim_cfg = cfg['optimizer']
    optim_type = optim_cfg['type']
    
    if optim_type == 'adam':
        optimizer = optim.Adam(
            model_params,
            lr=optim_cfg.get('lr', 1e-4),
            weight_decay=optim_cfg.get('weight_decay', 1e-5)
        )
    
    elif optim_type == 'sgd':
        optimizer = optim.SGD(
            model_params,
            lr=optim_cfg.get('lr', 1e-3),
            momentum=optim_cfg.get('momentum', 0.9),
            weight_decay=optim_cfg.get('weight_decay', 1e-5)
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")
    
    return optimizer


def get_scheduler(cfg, optimizer):
    """Get learning rate scheduler based on configuration."""
    if 'scheduler' not in cfg:
        return None
    
    sched_cfg = cfg['scheduler']
    sched_type = sched_cfg['type']
    
    if sched_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get('step_size', 10),
            gamma=sched_cfg.get('gamma', 0.1)
        )
    
    elif sched_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get('T_max', 10),
            eta_min=sched_cfg.get('eta_min', 0)
        )
    
    elif sched_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_cfg.get('factor', 0.1),
            patience=sched_cfg.get('patience', 10)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")
    
    return scheduler


def get_loss_fn(cfg):
    """Get loss function based on configuration."""
    loss_cfg = cfg.get('loss', {'type': 'dice'})
    loss_type = loss_cfg['type']
    
    if loss_type == 'dice':
        return dice_loss
    
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    
    elif loss_type == 'bce_dice':
        bce = nn.BCEWithLogitsLoss()
        def bce_dice_loss(pred, target):
            return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)
        return bce_dice_loss
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    valid_batches = 0  # Count valid batches
    
    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Training")):
        # Skip empty batches
        if images.numel() == 0 or masks.numel() == 0:
            continue
            
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        valid_batches += 1
    
    # Calculate loss only if we had valid batches
    if valid_batches > 0:
        epoch_loss = running_loss / valid_batches
    else:
        epoch_loss = float('inf')  # Indicate problem if no valid batches
        
    return epoch_loss


def validate(model, dataloader, loss_fn, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    dice_scores = []
    valid_batches = 0  # Count valid batches
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Validating")):
            # Skip empty batches
            if images.numel() == 0 or masks.numel() == 0:
                continue
                
            # Debug info for first batch
            if batch_idx == 0:
                print(f"Validation batch shape: images={images.shape}, masks={masks.shape}")
                print(f"Image range: [{images.min().item()}, {images.max().item()}]")
                print(f"Mask range: [{masks.min().item()}, {masks.max().item()}], unique values: {torch.unique(masks)}")
            
            # Normalize images to [0,1] if needed
            if images.max() > 1.0:
                images = images / 255.0
                
            # Ensure masks are binary
            if masks.max() > 1.0:
                masks = (masks > 0).float()
                
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle potential NaN outputs
            if torch.isnan(outputs).any():
                print(f"Warning: NaN values detected in model outputs at batch {batch_idx}")
                continue
            
            # Get binary predictions for visualization
            if batch_idx == 0:
                preds = (torch.sigmoid(outputs.detach()) > 0.5).float()
                pos_pixels = preds.sum().item()
                mask_pixels = masks.sum().item()
                print(f"Prediction positive pixels: {pos_pixels}, Mask positive pixels: {mask_pixels}")
                
            # Calculate loss
            try:
                loss = loss_fn(outputs, masks)
                
                # Calculate dice but filter out unreasonable values
                dice = dice_coef(outputs, masks)
                
                # Skip perfect scores if not justified by data
                if dice > 0.99 and masks.sum() > 10:  # Only if mask has sufficient positive pixels
                    print(f"Warning: Unusually high Dice score ({dice:.4f}) at batch {batch_idx}")
                    
                dice_scores.append(dice)
                running_loss += loss.item() * images.size(0)
                valid_batches += 1
                
                # Print distribution of dice scores periodically
                if (len(dice_scores) % 10 == 0) or (batch_idx + 1 == len(dataloader)):
                    print(f"Dice scores so far: min={min(dice_scores):.4f}, max={max(dice_scores):.4f}, "
                          f"mean={sum(dice_scores)/len(dice_scores):.4f}")
                    
            except Exception as e:
                print(f"Error during validation at batch {batch_idx}: {e}")
                continue
    
    # Calculate metrics only if we had valid batches
    if valid_batches > 0:
        epoch_loss = running_loss / valid_batches
        epoch_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    else:
        epoch_loss = float('inf')  # Indicate problem if no valid batches
        epoch_dice = 0.0
    
    return epoch_loss, epoch_dice


def save_checkpoint(model, optimizer, epoch, val_dice, val_loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
        'val_loss': val_loss,
    }
    
    # Save regular checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Save best model if this is the best so far
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pt'))


def run_training(cfg_path):
    """Run training based on configuration."""
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set random seed
    set_seed(cfg['seed'])
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg['output_dir'], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data configuration
    data_cfg = cfg['data']
    
    # Create datasets
    train_transforms = get_transforms(mode='train')
    val_transforms = get_transforms(mode='val')
    
    train_dataset = BrainMRIDataset(
        split_csv=data_cfg['split_csv'],
        data_dir=data_cfg['interim_dir'],
        split='train',
        transform=train_transforms,
        in_channels=cfg['model']['in_channels']
    )
    
    val_dataset = BrainMRIDataset(
        split_csv=data_cfg['split_csv'],
        data_dir=data_cfg['interim_dir'],
        split='val',
        transform=val_transforms,
        in_channels=cfg['model']['in_channels']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=custom_collate
    )
    
    # Build model
    model, freeze_epochs = get_model(cfg)
    model = model.to(device)
    
    # Loss function
    loss_fn = get_loss_fn(cfg)
    
    # Tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Training configuration
    epochs = cfg['epochs']
    
    # Metrics tracking
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_epoch = 0
    patience = cfg.get('patience', 15)  # Early stopping patience
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Check if we should unfreeze
        if freeze_epochs > 0 and epoch > freeze_epochs:
            print(f"Unfreezing encoder at epoch {epoch}")
            for param in model.parameters():
                param.requires_grad = True
        
        # Optimize different parameters based on freezing
        if freeze_epochs > 0 and epoch <= freeze_epochs:
            optimizer = get_optimizer(cfg, filter(lambda p: p.requires_grad, model.parameters()))
        else:
            optimizer = get_optimizer(cfg, model.parameters())
        
        # Get scheduler
        scheduler = get_scheduler(cfg, optimizer)
        
        # Train epoch
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)
        
        # Check for NaN or unrealistic values in validation metrics
        if val_dice > 0.99:
            print(f"WARNING: Very high Dice score detected ({val_dice:.4f}). Potential data leakage or processing issue.")
        
        if torch.isnan(torch.tensor(val_loss)) or torch.isnan(torch.tensor(val_dice)):
            print("WARNING: NaN values detected in validation metrics. Skipping checkpoint and continuing.")
            continue
            
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        
        # Check if model improved
        is_best = False
        
        # Track based on validation loss for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            is_best = True
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
        
        # Track best dice separately
        if val_dice > best_val_dice:
            best_val_dice = val_dice
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_dice=val_dice,
            val_loss=val_loss,
            checkpoint_dir=os.path.join(output_dir, 'checkpoints'),
            is_best=is_best
        )
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            print(f"Best model was at epoch {best_epoch} with val_loss={best_val_loss:.4f} and val_dice={best_val_dice:.4f}")
            break
    
    # Log final results
    final_metrics = {
        'best_val_loss': float(best_val_loss),
        'best_val_dice': float(best_val_dice),
        'best_epoch': best_epoch,
        'total_epochs': epoch
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print(f"Training completed. Best val_loss: {best_val_loss:.4f}, Best val_dice: {best_val_dice:.4f}")
    
    return best_val_dice


def cli():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--cfg", required=True, help="Path to config YAML file")
    
    args = parser.parse_args()
    run_training(args.cfg)


if __name__ == "__main__":
    cli() 