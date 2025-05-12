"""Dataset classes and utilities for brain MRI segmentation."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
import random

def create_dataframe(data_dir):
    """
    Create a DataFrame with image and mask pairs.
    
    Args:
        data_dir (str): Directory containing images and masks
    
    Returns:
        pd.DataFrame: DataFrame with image and mask paths
    """
    masks_paths = glob(os.path.join(data_dir, '*', '*_mask*'))
    images_paths = [mask_path.replace('_mask', '') for mask_path in masks_paths]
    
    df = pd.DataFrame(data={'image_path': images_paths, 'mask_path': masks_paths})
    return df

def split_dataframe(df, val_size=0.2, test_size=0.1, random_state=42):
    """
    Split DataFrame into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): DataFrame with image and mask paths
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        random_state (int): Random seed
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Get unique patients (assuming directory structure is data_dir/patient_id/...)
    df['patient_id'] = df['image_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    
    # Get unique patient IDs
    patient_ids = df['patient_id'].unique()
    
    # Shuffle patient IDs
    np.random.seed(random_state)
    np.random.shuffle(patient_ids)
    
    # Calculate split indices
    n_patients = len(patient_ids)
    n_val = int(n_patients * val_size)
    n_test = int(n_patients * test_size)
    n_train = n_patients - n_val - n_test
    
    # Split patient IDs
    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train+n_val]
    test_patients = patient_ids[n_train+n_val:]
    
    # Create dataframes
    train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} samples from {len(train_patients)} patients")
    print(f"Val: {len(val_df)} samples from {len(val_patients)} patients")
    print(f"Test: {len(test_df)} samples from {len(test_patients)} patients")
    
    return train_df, val_df, test_df


class BrainMRISegmentationDataset(Dataset):
    """Dataset for brain MRI segmentation with pairs of images and masks."""
    
    def __init__(self, dataframe, transform=None, augment=False, in_channels=3):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with image and mask paths
            transform (callable, optional): Transform to apply to image-mask pairs
            augment (bool): Whether to use data augmentation
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        """
        self.dataframe = dataframe
        self.transform = transform
        self.augment = augment
        self.in_channels = in_channels
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get paths
        img_path = self.dataframe.iloc[idx]['image_path']
        mask_path = self.dataframe.iloc[idx]['mask_path']
        
        # Load image
        try:
            image = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
        except Exception as e:
            print(f"Error loading image or mask: {e}")
            # Return a placeholder if loading fails
            if self.in_channels == 1:
                return torch.zeros(1, 256, 256), torch.zeros(1, 256, 256)
            else:
                return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256)
        
        # Convert RGB to grayscale if needed and in_channels=1
        if self.in_channels == 1 and len(image.shape) == 3 and image.shape[2] == 3:
            # Use proper RGB to grayscale conversion weights
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        elif self.in_channels == 3 and len(image.shape) == 2:
            # Convert grayscale to RGB
            image = np.stack([image, image, image], axis=2)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).astype(np.float32)
        
        # Normalize image to [0, 1] if not already
        if image.max() > 1.0:
            image = image / 255.0
            
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        # Add channel dimension if not present
        if image.ndim == 2:
            image = image.unsqueeze(0)  # (1, H, W)
        elif image.ndim == 3 and image.shape[2] in [1, 3]:  # HWC format
            image = image.permute(2, 0, 1)  # Convert to CHW format
            
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)
        elif mask.ndim == 3 and mask.shape[2] == 1:  # HWC format
            mask = mask.permute(2, 0, 1)  # Convert to CHW format
        
        # Apply transformations
        if self.transform:
            image, mask = self.transform(image, mask)
            
        return image, mask 