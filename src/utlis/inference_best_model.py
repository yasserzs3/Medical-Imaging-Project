#!/usr/bin/env python

"""
Inference script to load the best model weights from a training run 
and run inference on random seed images.
"""

import os
import sys
import json
import argparse
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import from the updated module structure
from src.data.dataset import create_dataframe, split_dataframe, BrainMRISegmentationDataset
from src.data.transforms import get_transforms
from src.models.unet import build_unet, UNet
from src.models.metrics import calculate_metrics


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


def find_best_model_dir(outputs_dir):
    """Find the best model directory based on the training_metrics.json file."""
    best_metric = -1
    best_dir = None
    
    # List all subdirectories in the outputs directory
    run_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    
    if not run_dirs:
        raise ValueError(f"No run directories found in {outputs_dir}")
    
    print(f"Found {len(run_dirs)} run directories: {run_dirs}")
    
    for run_dir in run_dirs:
        metrics_file = os.path.join(outputs_dir, run_dir, 'training_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Calculate the best combined metric (Dice + IoU) / 2
            best_run_metric = max([(dice + iou) / 2 for dice, iou in zip(metrics['val_dice'], metrics['val_iou'])])
            
            if best_run_metric > best_metric:
                best_metric = best_run_metric
                best_dir = os.path.join(outputs_dir, run_dir)
    
    if best_dir is None:
        # If no metrics.json found, just use the latest directory
        run_dirs.sort(reverse=True)  # Sort by timestamp (newest first)
        best_dir = os.path.join(outputs_dir, run_dirs[0])
        print(f"No metrics file found. Using latest run directory: {best_dir}")
    else:
        print(f"Best run directory: {best_dir} with combined metric of {best_metric:.4f}")
    
    return best_dir


def load_best_model(model_dir, device):
    """Load the best model from the specified directory."""
    model_path = os.path.join(model_dir, 'best_model.pt')
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found at {model_path}")
    
    # Load model args to get the model configuration
    args_path = os.path.join(model_dir, 'args.json')
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            args = json.load(f)
        in_channels = args.get('in_channels', 1)
    else:
        print("Args file not found. Using default in_channels=1")
        in_channels = 1
    
    # Use the original feature sizes [32, 64, 128, 256] that match the checkpoint
    # instead of the default [36, 72, 144, 288]
    original_features = [32, 64, 128, 256]
    
    # Create UNet directly with the original feature sizes
    model = UNet(
        in_channels=in_channels,
        n_classes=1,
        bilinear=True,
        features=original_features,
        dropout_rate=0.2
    )
    
    print(f"Created UNet model with {in_channels} input channels and features {original_features}")
    
    # Load the model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print model information
    print(f"Loaded model from epoch {checkpoint['epoch']} with:")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val Dice: {checkpoint['val_dice']:.4f}")
    print(f"  Val IoU: {checkpoint['val_iou']:.4f}")
    
    return model, in_channels


def run_inference(model, dataset, indices, device, output_dir, threshold=0.5):
    """Run inference on the specified image indices and visualize results."""
    results = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with 3 columns (Original, Ground Truth, Prediction)
    for idx in indices:
        image, mask = dataset[idx]
        
        # Move to device
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        mask = mask.unsqueeze(0).to(device)    # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            output = model(image)
            prediction = torch.sigmoid(output)
            binary_pred = (prediction > threshold).float()
        
        # Calculate metrics
        metrics = calculate_metrics(output, mask)
        
        # Move back to CPU for visualization
        image = image.cpu().squeeze()
        mask = mask.cpu().squeeze()
        prediction = prediction.cpu().squeeze()
        binary_pred = binary_pred.cpu().squeeze()
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[0] == 1):
            # Grayscale image (H,W) or (1,H,W)
            axes[0].imshow(image.squeeze(), cmap='gray')
        else:
            # RGB image (3,H,W)
            axes[0].imshow(image.permute(1, 2, 0))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Plot ground truth mask
        axes[1].imshow(mask.squeeze(), cmap='viridis')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        # Plot prediction with probability heatmap
        im = axes[2].imshow(prediction.squeeze(), cmap='plasma', vmin=0, vmax=1)
        axes[2].contour(binary_pred.squeeze(), levels=[0.5], colors='r', linestyles='solid')
        axes[2].set_title("Prediction (Probability)")
        axes[2].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Add metrics as text
        plt.figtext(0.5, 0.01, 
                  f"Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}", 
                  ha="center", fontsize=10, 
                  bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"inference_sample_{idx}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        results.append({
            'index': idx,
            'metrics': metrics
        })
    
    return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with the best model')
    parser.add_argument('--data_dir', type=str, default="data/raw/lgg-mri-segmentation/kaggle_3m",
                        help='Directory containing the MRI scans and masks')
    parser.add_argument('--outputs_dir', type=str, default="outputs",
                        help='Directory containing model outputs from training runs')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Specific model directory to use (default: find best model automatically)')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save the inference results')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='Number of random samples for inference. Set to 0 to use entire test dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    return parser.parse_args()


def main():
    """Main function for inference."""
    args = parse_arguments()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Print arguments
    print("Inference parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Determine model directory
    if args.model_dir is None:
        model_dir = find_best_model_dir(args.outputs_dir)
    else:
        model_dir = args.model_dir
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, in_channels = load_best_model(model_dir, device)
    
    # Determine data directory
    if not os.path.exists(args.data_dir):
        # Try to find the data directory relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        possible_data_dirs = [
            args.data_dir,  # Try as provided
            os.path.join(project_root, args.data_dir),  # Try relative to project root
            os.path.join(project_root, 'data', 'raw', 'lgg-mri-segmentation', 'kaggle_3m'),
            os.path.join(project_root, 'data', 'lgg-mri-segmentation', 'kaggle_3m'),
            os.path.join(project_root, 'data', 'raw'),
        ]
        
        for data_dir in possible_data_dirs:
            if os.path.exists(data_dir):
                args.data_dir = data_dir
                print(f"Found data directory: {args.data_dir}")
                break
        
        if not os.path.exists(args.data_dir):
            raise ValueError("Data directory not found. Please specify a valid --data_dir.")
    
    # Create dataframe from the data directory
    print(f"Creating dataframe from {args.data_dir}...")
    df = create_dataframe(args.data_dir)
    print(f"DataFrame created with {len(df)} entries.")
    
    # Split into train, validation, and test sets
    _, _, test_df = split_dataframe(df, val_size=0.2, test_size=0.1, random_state=args.seed)
    
    # Create dataset without augmentations for visualization
    test_transforms = get_transforms(mode='val')
    test_dataset = BrainMRISegmentationDataset(
        dataframe=test_df,
        transform=test_transforms,
        augment=False,
        in_channels=in_channels
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Select indices for inference
    if args.sample_size > 0:
        # Use a random sample
        if args.sample_size > len(test_dataset):
            print(f"Warning: sample_size ({args.sample_size}) is greater than dataset size ({len(test_dataset)}). Using entire dataset.")
            indices = list(range(len(test_dataset)))
        else:
            print(f"Running inference on {args.sample_size} random samples...")
            indices = random.sample(range(len(test_dataset)), args.sample_size)
    else:
        # Use the entire test dataset
        print(f"Running inference on the entire test dataset ({len(test_dataset)} samples)...")
        indices = list(range(len(test_dataset)))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    print("Running inference...")
    results = run_inference(model, test_dataset, indices, device, args.output_dir, args.threshold)
    
    # Compute and print average metrics
    avg_metrics = {
        'dice': np.mean([r['metrics']['dice'] for r in results]),
        'f1': np.mean([r['metrics']['f1'] for r in results]),
        'iou': np.mean([r['metrics']['iou'] for r in results]),
        'accuracy': np.mean([r['metrics']['accuracy'] for r in results]),
        'precision': np.mean([r['metrics']['precision'] for r in results]),
        'recall': np.mean([r['metrics']['recall'] for r in results]),
        'specificity': np.mean([r['metrics']['specificity'] for r in results])
    }
    
    print("\nAverage Metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save metrics to JSON
    with open(os.path.join(args.output_dir, 'inference_metrics.json'), 'w') as f:
        json.dump({
            'results': results,
            'average_metrics': avg_metrics,
            'model_dir': model_dir,
            'args': vars(args)
        }, f, indent=4, default=str)
    
    print(f"Inference results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 