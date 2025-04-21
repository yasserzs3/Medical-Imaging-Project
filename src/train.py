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
    pred = torch.sigmoid(pred) > threshold
    pred = pred.float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
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
                
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            # Calculate metrics
            dice = dice_coef(outputs, masks)
            dice_scores.append(dice)
            
            running_loss += loss.item() * images.size(0)
            valid_batches += 1
    
    # Calculate metrics only if we had valid batches
    if valid_batches > 0 and dice_scores:
        epoch_loss = running_loss / valid_batches
        epoch_dice = np.mean(dice_scores)
    else:
        epoch_loss = float('inf')
        epoch_dice = 0.0  # Indicate problem if no valid batches
    
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
    """Main training function."""
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Get experiment name and create directories
    exp_name = os.path.splitext(os.path.basename(cfg_path))[0]
    output_dir = os.path.join(cfg.get('output_dir', 'outputs'), exp_name)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    log_dir = os.path.join(output_dir, 'logs')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir)
    
    # Set random seed
    set_seed(cfg.get('seed', 42))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    data_dir = cfg['data']['interim_dir']
    split_csv = cfg['data']['split_csv']
    
    # Get model's input channel requirements
    in_channels = cfg['model'].get('in_channels', 1)
    
    # Re-enable transforms
    train_transforms = get_transforms('train')
    val_transforms = get_transforms('val')
    
    train_dataset = BrainMRIDataset(
        split_csv=split_csv,
        data_dir=data_dir,
        split='train',
        transform=train_transforms,
        in_channels=in_channels
    )
    
    val_dataset = BrainMRIDataset(
        split_csv=split_csv,
        data_dir=data_dir,
        split='val',
        transform=val_transforms,
        in_channels=in_channels
    )
    
    # Create dataloaders
    batch_size = cfg.get('batch_size', 8)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.get('num_workers', 4),
        collate_fn=custom_collate
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get('num_workers', 4),
        collate_fn=custom_collate
    )
    
    # Build model
    model, freeze_epochs = get_model(cfg)
    model = model.to(device)
    
    # Get optimizer and scheduler
    optimizer = get_optimizer(cfg, model.parameters())
    scheduler = get_scheduler(cfg, optimizer)
    
    # Get loss function
    loss_fn = get_loss_fn(cfg)
    
    # Initialize training variables
    num_epochs = cfg.get('epochs', 50)
    best_dice = 0.0
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Unfreeze encoder if needed
        if freeze_epochs > 0 and epoch > freeze_epochs and hasattr(model, 'unfreeze_encoder'):
            print("Unfreezing encoder")
            model.unfreeze_encoder()
            optimizer = get_optimizer(cfg, model.parameters())
            if scheduler:
                scheduler = get_scheduler(cfg, optimizer)
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_dice = validate(model, val_dataloader, loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_dice=val_dice,
            val_loss=val_loss,
            checkpoint_dir=checkpoint_dir,
            is_best=is_best
        )
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    
    # Training finished
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {str(datetime.timedelta(seconds=int(elapsed_time)))}")
    print(f"Best validation Dice coefficient: {best_dice:.4f}")
    
    # Save training summary
    summary = {
        'exp_name': exp_name,
        'best_dice': float(best_dice),
        'train_time': float(elapsed_time),
        'completed_epochs': num_epochs,
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    writer.close()


def cli():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--cfg", required=True, help="Path to config YAML file")
    
    args = parser.parse_args()
    run_training(args.cfg)


if __name__ == "__main__":
    cli() 