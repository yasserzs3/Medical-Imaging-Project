import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from tqdm import tqdm
import yaml
import glob
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BrainMRIDataset, get_transforms
from src.models import unet_scratch, unet_resnet34


def load_model(checkpoint_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device (torch.device): Device to load model to
    
    Returns:
        nn.Module: Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint file path
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Build model
    model_cfg = cfg['model']
    model_type = model_cfg['type']
    
    if model_type == 'unet_scratch':
        model = unet_scratch.build(
            in_channels=model_cfg.get('in_channels', 1),
            n_classes=model_cfg.get('n_classes', 1),
            bilinear=model_cfg.get('bilinear', True)
        )
    
    elif model_type == 'unet_resnet34':
        model, _ = unet_resnet34.build(
            in_channels=model_cfg.get('in_channels', 3),
            n_classes=model_cfg.get('n_classes', 1),
            pretrained=False  # No need for pretrained during inference
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path} (val_dice: {checkpoint.get('val_dice', 'N/A'):.4f})")
    
    return model, cfg


def predict_slice(model, image, device, threshold=0.5):
    """
    Make prediction for a single image slice.
    
    Args:
        model (nn.Module): Trained model
        image (torch.Tensor): Input image tensor [C, H, W]
        device (torch.device): Device to run inference on
        threshold (float): Threshold for binary prediction
    
    Returns:
        np.ndarray: Predicted binary mask
        np.ndarray: Predicted probability mask
    """
    # Add batch dimension
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        
    # Convert to probability [0, 1]
    probs = torch.sigmoid(output).cpu().numpy().squeeze()
    
    # Apply threshold for binary prediction
    pred_mask = (probs > threshold).astype(np.uint8)
    
    return pred_mask, probs


def save_prediction_overlay(image, mask, pred, output_path, alpha=0.5):
    """
    Save an overlay of the image, ground truth, and prediction.
    
    Args:
        image (np.ndarray): Input image [H, W]
        mask (np.ndarray): Ground truth mask [H, W]
        pred (np.ndarray): Predicted mask [H, W]
        output_path (str): Path to save the overlay
        alpha (float): Transparency for the overlay
    """
    # Create RGB versions of masks
    red_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    red_mask[mask > 0] = [255, 0, 0]  # Ground truth in red
    
    blue_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    blue_mask[pred > 0] = [0, 0, 255]  # Prediction in blue
    
    # Create a purple mask for correct predictions (overlap)
    purple_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    purple_mask[(mask > 0) & (pred > 0)] = [255, 0, 255]  # Overlap in purple
    
    # Convert grayscale image to RGB
    img_rgb = np.stack([image, image, image], axis=2)
    
    # Normalize to [0, 255] if not already
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    else:
        img_rgb = img_rgb.astype(np.uint8)
    
    # Create overlay
    overlay = img_rgb.copy()
    
    # Add masks to overlay with alpha blending
    idx_gt = (mask > 0) & (pred == 0)  # Ground truth only (FN)
    idx_pred = (mask == 0) & (pred > 0)  # Prediction only (FP)
    idx_both = (mask > 0) & (pred > 0)  # Both (TP)
    
    overlay[idx_gt] = (1 - alpha) * img_rgb[idx_gt] + alpha * red_mask[idx_gt]
    overlay[idx_pred] = (1 - alpha) * img_rgb[idx_pred] + alpha * blue_mask[idx_pred]
    overlay[idx_both] = (1 - alpha) * img_rgb[idx_both] + alpha * purple_mask[idx_both]
    
    # Save the overlay
    Image.fromarray(overlay).save(output_path)


def run_inference(checkpoint_path, output_dir=None, save_logits=False, threshold=0.5):
    """
    Run inference on validation and test datasets.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        output_dir (str): Directory to save predictions
        save_logits (bool): Whether to save raw logits
        threshold (float): Threshold for binary prediction
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and config
    model, cfg = load_model(checkpoint_path, device)
    
    # Create output directory if not specified
    if output_dir is None:
        exp_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_dir = os.path.join('outputs', exp_name, 'predictions', checkpoint_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets
    data_dir = cfg['data']['interim_dir']
    split_csv = cfg['data']['split_csv']
    
    for split in ['val', 'test']:
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Create dataset
        dataset = BrainMRIDataset(
            split_csv=split_csv,
            data_dir=data_dir,
            split=split,
            transform=get_transforms('val'),
            return_path=True
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        
        # Run inference
        metrics = {'dice': [], 'precision': [], 'recall': []}
        start_time = time.time()
        
        for i, (image, mask, path) in enumerate(tqdm(dataloader, desc=f"Inferring on {split}")):
            # Get filename
            filename = os.path.basename(path[0])
            patient_id = os.path.basename(os.path.dirname(path[0]))
            
            # Create patient directory
            patient_dir = os.path.join(split_output_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            # Make prediction
            pred_mask, probs = predict_slice(model, image[0], device, threshold)
            
            # Save prediction overlay
            image_np = image[0].numpy().squeeze()
            mask_np = mask[0].numpy().squeeze()
            
            overlay_path = os.path.join(patient_dir, f"{os.path.splitext(filename)[0]}_overlay.png")
            save_prediction_overlay(image_np, mask_np, pred_mask, overlay_path)
            
            # Save logits if requested
            if save_logits:
                logits_path = os.path.join(patient_dir, f"{os.path.splitext(filename)[0]}_logits.npy")
                np.save(logits_path, probs)
            
            # Calculate metrics
            if mask_np.sum() > 0 or pred_mask.sum() > 0:
                # Calculate Dice
                intersection = np.sum(pred_mask * mask_np)
                dice = (2.0 * intersection) / (np.sum(pred_mask) + np.sum(mask_np) + 1e-6)
                
                # Calculate precision and recall
                true_pos = np.sum(pred_mask * mask_np)
                false_pos = np.sum(pred_mask * (1 - mask_np))
                false_neg = np.sum((1 - pred_mask) * mask_np)
                
                precision = true_pos / (true_pos + false_pos + 1e-6)
                recall = true_pos / (true_pos + false_neg + 1e-6)
                
                metrics['dice'].append(dice)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
        
        # Calculate average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        
        # Print and save metrics
        print(f"\n{split.upper()} set metrics:")
        print(f"Dice: {avg_metrics['dice']:.4f}")
        print(f"Precision: {avg_metrics['precision']:.4f}")
        print(f"Recall: {avg_metrics['recall']:.4f}")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f"{split}_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Dice: {avg_metrics['dice']:.4f}\n")
            f.write(f"Precision: {avg_metrics['precision']:.4f}\n")
            f.write(f"Recall: {avg_metrics['recall']:.4f}\n")
            f.write(f"Inference time: {time.time() - start_time:.2f} seconds\n")
    
    print(f"\nPredictions saved to {output_dir}")


def predict_volume(checkpoint_path, volume_dir, output_dir=None, threshold=0.5):
    """
    Run inference on a single volume (set of slices).
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        volume_dir (str): Directory containing volume slices
        output_dir (str): Directory to save predictions
        threshold (float): Threshold for binary prediction
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, _ = load_model(checkpoint_path, device)
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(volume_dir), 'predictions')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all slices
    slice_paths = sorted(glob.glob(os.path.join(volume_dir, '*.npy')))
    
    if not slice_paths:
        print(f"No .npy files found in {volume_dir}")
        return
    
    # Run inference on each slice
    for slice_path in tqdm(slice_paths, desc="Predicting volume"):
        # Load slice
        data = np.load(slice_path, allow_pickle=True).item()
        image = data['image']
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        
        # Make prediction
        pred_mask, probs = predict_slice(model, image_tensor, device, threshold)
        
        # Save prediction
        filename = os.path.basename(slice_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pred.png")
        
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(image, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(pred_mask, cmap='gray')
        plt.title(f'Prediction (t={threshold})')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(probs, cmap='jet')
        plt.colorbar()
        plt.title('Probability Map')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    print(f"\nPredictions saved to {output_dir}")


def cli():
    """Command-line interface for prediction."""
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", help="Directory to save predictions")
    parser.add_argument("--save_logits", action="store_true", help="Save raw logits")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary prediction")
    parser.add_argument("--volume", help="Path to volume directory for single volume prediction")
    
    args = parser.parse_args()
    
    if args.volume:
        predict_volume(args.checkpoint, args.volume, args.output_dir, args.threshold)
    else:
        run_inference(args.checkpoint, args.output_dir, args.save_logits, args.threshold)


if __name__ == "__main__":
    cli() 