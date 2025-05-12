#!/usr/bin/env python

import os
import argparse
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

# Import from the updated module structure
from src.data.dataset import create_dataframe, split_dataframe, BrainMRISegmentationDataset
from src.data.transforms import get_transforms
from src.models.unet import build_unet
from src.models.losses import dice_loss, iou_loss, ComboLoss, dice_coef, iou_score
from src.models.metrics import ConfusionMatrix


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


def create_datasets(args):
    """Create train and validation datasets."""
    # Create dataframe with image and mask pairs
    df = create_dataframe(args.data_dir)
    print(f"Found {len(df)} image-mask pairs")
    
    # Split into train, validation, and test sets
    train_df, val_df, _ = split_dataframe(df, val_size=0.2, test_size=0.1, random_state=args.seed)
    
    # Get transforms
    train_transform = get_transforms(mode='train') if args.augment else get_transforms(mode='val')
    val_transform = get_transforms(mode='val')
    
    # Create datasets
    train_dataset = BrainMRISegmentationDataset(
        dataframe=train_df,
        transform=train_transform,
        augment=args.augment,
        in_channels=args.in_channels
    )
    
    val_dataset = BrainMRISegmentationDataset(
        dataframe=val_df,
        transform=val_transform,
        augment=False,
        in_channels=args.in_channels
    )
    
    return train_dataset, val_dataset


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Training")):
        # Move to device
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
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, loss_fn, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    dice_scores = []
    iou_scores = []
    
    # Initialize confusion matrix
    conf_matrix = ConfusionMatrix(threshold=0.5)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Validating")):
            # Debug info for first batch
            if batch_idx == 0:
                print(f"Validation batch shape: images={images.shape}, masks={masks.shape}")
                print(f"Image range: [{images.min().item()}, {images.max().item()}]")
                print(f"Mask range: [{masks.min().item()}, {masks.max().item()}], unique values: {torch.unique(masks)}")
                print(f"Positive mask pixels: {(masks > 0).sum().item()}")
            
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = loss_fn(outputs, masks)
            running_loss += loss.item() * images.size(0)
            
            # Calculate metrics
            batch_dice = dice_coef(outputs, masks)
            batch_iou = iou_score(outputs, masks)
            dice_scores.append(batch_dice)
            iou_scores.append(batch_iou)
            
            # Update confusion matrix
            conf_matrix.update(outputs, masks)
            
            # Print predictions for first batch
            if batch_idx == 0:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                pos_pixels = preds.sum().item()
                mask_pixels = masks.sum().item()
                print(f"Prediction positive pixels: {pos_pixels}, Mask positive pixels: {mask_pixels}")
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = sum(dice_scores) / len(dice_scores)
    epoch_iou = sum(iou_scores) / len(iou_scores)
    
    # Get detailed metrics from confusion matrix
    detailed_metrics = conf_matrix.get_metrics()
    
    # Print detailed metrics
    print(f"Validation metrics:")
    print(f"  Accuracy: {detailed_metrics['accuracy']:.4f}")
    print(f"  Precision: {detailed_metrics['precision']:.4f}")
    print(f"  Recall: {detailed_metrics['recall']:.4f}")
    print(f"  Specificity: {detailed_metrics['specificity']:.4f}")
    print(f"  F1 Score: {detailed_metrics['f1']:.4f}")
    
    return epoch_loss, epoch_dice, epoch_iou, detailed_metrics


def train(args):
    """Train the model."""
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(args)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = build_unet(in_channels=args.in_channels, n_classes=1)
    print(f"Created UNet model with {args.in_channels} input channels")
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Create loss function (Combined loss: BCE + Dice + IoU)
    if args.loss_type == 'combo':
        loss_fn = ComboLoss(weights=[0.3, 0.4, 0.3])
        print("Using combined loss: BCE + Dice + IoU")
    elif args.loss_type == 'bce_dice':
        bce_loss = nn.BCEWithLogitsLoss()
        loss_fn = lambda pred, target: 0.5 * bce_loss(pred, target) + 0.5 * dice_loss(pred, target)
        print("Using BCE + Dice loss")
    elif args.loss_type == 'bce_iou':
        bce_loss = nn.BCEWithLogitsLoss()
        loss_fn = lambda pred, target: 0.5 * bce_loss(pred, target) + 0.5 * iou_loss(pred, target)
        print("Using BCE + IoU loss")
    elif args.loss_type == 'dice':
        loss_fn = dice_loss
        print("Using Dice loss")
    elif args.loss_type == 'iou':
        loss_fn = iou_loss
        print("Using IoU loss")
    else:  # 'bce'
        loss_fn = nn.BCEWithLogitsLoss()
        print("Using BCE loss")
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Initialize training variables
    best_combined = 0.0
    best_dice = 0.0
    best_iou = 0.0
    best_loss = float('inf')
    patience_counter = 0
    
    # Initialize metrics tracking dictionary for JSON
    metrics_history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_specificity': [],
        'val_f1': [],
        'lr': []
    }
    
    # Path for metrics JSON file
    metrics_json_path = os.path.join(output_dir, 'training_metrics.json')
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_dice, val_iou, detailed_metrics = validate(model, val_loader, loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics history dictionary
        metrics_history['epochs'].append(epoch)
        metrics_history['train_loss'].append(float(train_loss))
        metrics_history['val_loss'].append(float(val_loss))
        metrics_history['val_dice'].append(float(val_dice))
        metrics_history['val_iou'].append(float(val_iou))
        metrics_history['val_accuracy'].append(float(detailed_metrics['accuracy']))
        metrics_history['val_precision'].append(float(detailed_metrics['precision']))
        metrics_history['val_recall'].append(float(detailed_metrics['recall']))
        metrics_history['val_specificity'].append(float(detailed_metrics['specificity']))
        metrics_history['val_f1'].append(float(detailed_metrics['f1']))
        metrics_history['lr'].append(float(current_lr))
        
        # Save updated metrics to JSON file after each epoch
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics_history, f, indent=4)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('Accuracy/val', detailed_metrics['accuracy'], epoch)
        writer.add_scalar('Precision/val', detailed_metrics['precision'], epoch)
        writer.add_scalar('Recall/val', detailed_metrics['recall'], epoch)
        writer.add_scalar('Specificity/val', detailed_metrics['specificity'], epoch)
        writer.add_scalar('F1/val', detailed_metrics['f1'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Calculate combined metric (Dice + IoU) / 2
        combined_metric = (val_dice + val_iou) / 2
        
        # Check if model improved
        if combined_metric > best_combined:
            best_combined = combined_metric
            best_dice = val_dice
            best_iou = val_iou
            best_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'combined_metric': combined_metric,
            }, os.path.join(output_dir, 'best_model.pt'))
            
            print(f"Saved new best model with Combined: {best_combined:.4f} (Dice: {best_dice:.4f}, IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            print(f"Combined metric did not improve, patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'combined_metric': combined_metric,
        }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_dice': val_dice,
        'val_iou': val_iou,
        'combined_metric': combined_metric,
    }, os.path.join(output_dir, 'final_model.pt'))
    
    # Save final results
    results = {
        'best_combined': float(best_combined),
        'best_dice': float(best_dice),
        'best_iou': float(best_iou),
        'best_loss': float(best_loss),
        'final_dice': float(val_dice),
        'final_iou': float(val_iou),
        'final_loss': float(val_loss),
        'final_accuracy': float(detailed_metrics['accuracy']),
        'final_precision': float(detailed_metrics['precision']),
        'final_recall': float(detailed_metrics['recall']),
        'final_specificity': float(detailed_metrics['specificity']),
        'final_f1': float(detailed_metrics['f1']),
        'epochs': epoch
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTraining completed. Best Combined: {best_combined:.4f}")
    print(f"Results saved to {output_dir}")
    
    return best_combined


def main():
    parser = argparse.ArgumentParser(description="Train UNet for brain MRI segmentation")
    parser.add_argument("--data_dir", type=str, default="data/raw/lgg-mri-segmentation/kaggle_3m", help="Directory containing image and mask files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save models and logs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.00005, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels (1 for grayscale, 3 for RGB)")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--loss_type", type=str, default="combo", 
                      choices=["bce", "dice", "iou", "bce_dice", "bce_iou", "combo"], 
                      help="Loss function to use")
    
    args = parser.parse_args()
    
    print("Starting training with the following parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    train(args)


if __name__ == "__main__":
    main() 