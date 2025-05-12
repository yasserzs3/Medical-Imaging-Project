"""
Script to generate and save plots for random MRI scans, masks, and overlaid images.

This script selects random samples from the brain MRI dataset and creates 
visualizations of the original scan, the segmentation mask, and an overlay
of both to highlight the tumor regions.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import argparse

# Adjust path to import from src
module_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import from the project's data module
from data.dataset import create_dataframe, BrainMRISegmentationDataset
from data.transforms import get_transforms

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate plots for brain MRI scans and masks')
    parser.add_argument('--data_dir', type=str, default="data/raw/lgg-mri-segmentation/kaggle_3m",
                        help='Directory containing the MRI scans and masks')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                        help='Directory to save the generated plots')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of random samples to visualize')
    parser.add_argument('--in_channels', type=int, default=1, choices=[1, 3],
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--show_augmentations', action='store_true',
                        help='Show augmented versions of the images')
    parser.add_argument('--num_augmentations', type=int, default=3,
                        help='Number of augmented versions to show per image')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation (matches training script parameter)')
    
    return parser.parse_args()

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def plot_image_mask_overlay(image, mask, title='', save_path=None):
    """
    Plot an image, its corresponding mask, and an overlay of both.
    
    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W]
        mask (torch.Tensor): Mask tensor of shape [1, H, W]
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    # Create a figure with 3 subplots (image, mask, overlay)
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # Prepare image for display
    img_display = image.permute(1, 2, 0).cpu().numpy()
    if img_display.shape[2] == 1:
        img_display = img_display.squeeze(axis=2)
        cmap_img = 'gray'
    else:
        cmap_img = None
    img_display = np.clip(img_display, 0, 1)
    
    # Prepare mask for display
    mask_display = mask.squeeze(0).cpu().numpy()
    
    # 1. Plot original image
    ax1 = plt.subplot(gs[0])
    ax1.imshow(img_display, cmap=cmap_img)
    ax1.set_title('MRI Scan')
    ax1.axis('off')
    
    # 2. Plot mask
    ax2 = plt.subplot(gs[1])
    ax2.imshow(mask_display, cmap='gray')
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    
    # 3. Plot overlay
    ax3 = plt.subplot(gs[2])
    ax3.imshow(img_display, cmap=cmap_img)
    
    # Create a custom colormap for the overlay (transparent to red)
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]  # transparent to semi-transparent red
    red_transparent_cmap = LinearSegmentedColormap.from_list("red_transparent", colors)
    
    # Plot mask overlay with the custom colormap
    ax3.imshow(mask_display, cmap=red_transparent_cmap, alpha=0.7)
    ax3.set_title('Overlay (Tumor Region)')
    ax3.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()

def plot_augmentations(original_image, original_mask, transform, num_augmentations=3, title='', save_path=None):
    """
    Plot original image/mask and multiple augmented versions.
    
    Args:
        original_image (torch.Tensor): Original image tensor
        original_mask (torch.Tensor): Original mask tensor
        transform: Transform function to apply
        num_augmentations (int): Number of augmented versions to show
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    # Create a figure with rows for original and each augmentation
    fig, axes = plt.subplots(num_augmentations + 1, 3, figsize=(15, 5 * (num_augmentations + 1)))
    
    # Function to plot a single row (image, mask, overlay)
    def plot_row(ax_row, img, msk, row_title):
        # Prepare image for display
        img_display = img.permute(1, 2, 0).cpu().numpy()
        if img_display.shape[2] == 1:
            img_display = img_display.squeeze(axis=2)
            cmap_img = 'gray'
        else:
            cmap_img = None
        img_display = np.clip(img_display, 0, 1)
        
        # Prepare mask for display
        mask_display = msk.squeeze(0).cpu().numpy()
        
        # Plot image
        ax_row[0].imshow(img_display, cmap=cmap_img)
        ax_row[0].set_title(f'{row_title} - Scan')
        ax_row[0].axis('off')
        
        # Plot mask
        ax_row[1].imshow(mask_display, cmap='gray')
        ax_row[1].set_title(f'{row_title} - Mask')
        ax_row[1].axis('off')
        
        # Plot overlay
        ax_row[2].imshow(img_display, cmap=cmap_img)
        
        # Create a custom colormap for the overlay
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]
        red_transparent_cmap = LinearSegmentedColormap.from_list("red_transparent", colors)
        
        # Plot mask overlay
        ax_row[2].imshow(mask_display, cmap=red_transparent_cmap, alpha=0.7)
        ax_row[2].set_title(f'{row_title} - Overlay')
        ax_row[2].axis('off')
    
    # Plot original image/mask
    plot_row(axes[0], original_image, original_mask, "Original")
    
    # Plot augmented versions
    for i in range(num_augmentations):
        # Apply transformations
        aug_image, aug_mask = transform(original_image.clone(), original_mask.clone())
        plot_row(axes[i+1], aug_image, aug_mask, f"Augmentation {i+1}")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()

def main():
    """Main function to generate and save plots."""
    args = parse_arguments()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Print arguments
    print("Visualization parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
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
    
    # Create output directory
    create_output_directory(args.output_dir)
    
    # Create dataframe from the data directory
    print(f"Creating dataframe from {args.data_dir}...")
    df = create_dataframe(args.data_dir)
    print(f"DataFrame created with {len(df)} entries.")
    
    # Get transforms
    train_transforms = get_transforms(mode='train')
    
    # Create dataset without augmentations for visualization
    vis_dataset = BrainMRISegmentationDataset(df, augment=False, in_channels=args.in_channels)
    print(f"Dataset size: {len(vis_dataset)}")
    
    # Generate plots for random samples
    print(f"Generating plots for {args.num_samples} random samples...")
    
    # Get unique indices to avoid duplicates
    dataset_size = len(vis_dataset)
    if args.num_samples > dataset_size:
        print(f"Warning: Requested {args.num_samples} samples but dataset only has {dataset_size}.")
        args.num_samples = dataset_size
    
    indices = random.sample(range(dataset_size), args.num_samples)
    
    for i, idx in enumerate(indices):
        # Get image and mask
        try:
            image, mask = vis_dataset[idx]
            
            # Generate plot title and save path
            title = f"Sample {i+1}/{args.num_samples} (Index: {idx})"
            
            if args.show_augmentations:
                # Show original and augmented versions
                aug_save_path = os.path.join(args.output_dir, f"sample_{idx}_augmentations.png")
                plot_augmentations(
                    image, mask, 
                    train_transforms, 
                    num_augmentations=args.num_augmentations,
                    title=title, 
                    save_path=aug_save_path
                )
            else:
                # Show just the original image/mask/overlay
                save_path = os.path.join(args.output_dir, f"sample_{idx}_visualization.png")
                plot_image_mask_overlay(image, mask, title=title, save_path=save_path)
            
            # Print progress
            print(f"Processed sample {i+1}/{args.num_samples}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
    
    print(f"Visualization complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 