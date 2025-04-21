import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BrainMRIDataset(Dataset):
    """Dataset for brain MRI segmentation."""
    
    def __init__(self, 
                 split_csv, 
                 data_dir, 
                 split='train', 
                 transform=None, 
                 preload=False,
                 return_path=False,
                 in_channels=1):
        """
        Args:
            split_csv (str): Path to CSV with patient splits
            data_dir (str): Directory with preprocessed .npy files
            split (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Transform to apply to image-mask pairs
            preload (bool): If True, load all data into memory
            return_path (bool): If True, return file path with each sample
            in_channels (int): Number of input channels (1 or 3)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.preload = preload
        self.return_path = return_path
        self.in_channels = in_channels
        
        # Load split information
        splits_df = pd.read_csv(split_csv)
        self.patient_ids = splits_df[splits_df['split'] == split]['patient_id'].tolist()
        
        # Get all slice paths for the patients in this split
        self.samples = []
        for patient_id in self.patient_ids:
            patient_dir = os.path.join(data_dir, patient_id)
            if os.path.exists(patient_dir):
                slice_files = [f for f in os.listdir(patient_dir) if f.endswith('.npy')]
                for slice_file in slice_files:
                    self.samples.append(os.path.join(patient_dir, slice_file))
        
        # Preload data if requested
        self.cache = {}
        if self.preload:
            for path in self.samples:
                self.cache[path] = np.load(path, allow_pickle=True).item()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        
        # Load sample (either from cache or disk)
        if path in self.cache:
            sample = self.cache[path]
        else:
            try:
                sample = np.load(path, allow_pickle=True).item()
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Return a dummy sample in case of errors
                return torch.zeros((self.in_channels, 256, 256)), torch.zeros((1, 256, 256))
        
        # Extract image and mask
        image = sample['image']  # Expecting shape (H, W) or (H, W, C)
        mask = sample['mask']    # Expecting shape (H, W)
        
        # Debug information
        if idx == 0:  # Print info for the first sample
            print(f"Original image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
            if mask is not None:
                print(f"Original mask shape: {mask.shape}, dtype: {mask.dtype}, range: [{mask.min()}, {mask.max()}]")
            else:
                print("Mask is None")
        
        # Handle RGB images - convert to grayscale properly
        if image.ndim > 2:
            # If image has RGB channels as last dimension (H, W, C)
            if image.shape[-1] in [3, 4]:  # RGB or RGBA
                # Use a proper RGB to grayscale conversion (weighted average)
                if image.shape[-1] == 3:  # RGB
                    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
                else:  # RGBA
                    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            # If channels are first dimension (C, H, W)
            elif image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                if image.shape[0] == 1:
                    image = image[0]  # Just take the first channel
                elif image.shape[0] == 3:  # RGB
                    image = np.dot(image.transpose(1, 2, 0)[..., :3], [0.2989, 0.5870, 0.1140])
                else:  # RGBA
                    image = np.dot(image.transpose(1, 2, 0)[..., :3], [0.2989, 0.5870, 0.1140])
        
        # Ensure mask is 2D
        if mask is not None and mask.ndim > 2:
            # For mask, we want to preserve positive values, so take max across channels
            if mask.shape[-1] in [3, 4]:  # Channels last
                mask = np.max(mask, axis=-1)
            elif mask.ndim == 3 and mask.shape[0] in [1, 3, 4]:  # Channels first
                mask = np.max(mask, axis=0)
        elif mask is None:
            # Create an empty mask if none exists
            mask = np.zeros_like(image)
        
        # Normalize image to [0, 1] range if it's not already
        if image.max() > 1.0:
            image = image / 255.0
        
        # Ensure mask is binary
        if mask is not None and np.max(mask) > 1.0:
            mask = (mask > 0).astype(np.float32)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        # Add channel dimension if not present
        if image.ndim == 2:
            image = image.unsqueeze(0)  # (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)    # (1, H, W)
        
        # Ensure consistent channel dimensions based on model requirements
        if self.in_channels == 3 and image.shape[0] == 1:
            # Convert single-channel to 3-channel for models like ResNet
            image = image.repeat(3, 1, 1)  # (3, H, W)
        
        # Apply transformations if any
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Final debug info after processing
        if idx == 0:  # Print info for the first sample
            print(f"Processed image shape: {image.shape}, range: [{image.min().item()}, {image.max().item()}]")
            print(f"Processed mask shape: {mask.shape}, range: [{mask.min().item()}, {mask.max().item()}]")
        
        if self.return_path:
            return image, mask, path
        return image, mask
    
    def to_rgb(self, image):
        """Convert single-channel image to 3-channel for pretrained models."""
        return image.repeat(3, 1, 1)


class TransformCompose:
    """Compose multiple transforms together for image-mask pairs."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_transforms(mode='train'):
    """Get transforms for training or validation/testing.
    
    Args:
        mode (str): 'train' or 'val'/'test'
    
    Returns:
        callable: Transformation function
    """
    if mode == 'train':
        return TransformCompose([
            RandomFlip(),
            RandomRotation(10),
            RandomBrightnessContrast(0.1, 0.1),
        ])
    else:
        return TransformCompose([])  # No augmentations for val/test


class RandomFlip:
    """Random horizontal or vertical flip."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if np.random.random() < self.p:
            # Horizontal flip
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if np.random.random() < self.p:
            # Vertical flip
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        return image, mask


class RandomRotation:
    """Random rotation within given degrees."""
    
    def __init__(self, degrees):
        self.degrees = degrees
    
    def __call__(self, image, mask):
        angle = np.random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        return image, mask


class RandomBrightnessContrast:
    """Random brightness and contrast adjustment."""
    
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast
    
    def __call__(self, image, mask):
        # Only apply to image, not mask
        if np.random.random() < 0.5:
            brightness_factor = np.random.uniform(1-self.brightness, 1+self.brightness)
            # Make sure image has the right shape and type for torchvision transforms
            if isinstance(image, torch.Tensor):
                # Check if channels dimension is first
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    image = transforms.functional.adjust_brightness(image, brightness_factor)
                else:
                    # Handle case where tensor doesn't have expected shape
                    image = image * brightness_factor
            else:
                image = image * brightness_factor
        
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(1-self.contrast, 1+self.contrast)
            # Make sure image has the right shape and type for torchvision transforms
            if isinstance(image, torch.Tensor):
                # Check if channels dimension is first
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    image = transforms.functional.adjust_contrast(image, contrast_factor)
                else:
                    # Apply manual contrast adjustment
                    mean = image.mean()
                    image = (image - mean) * contrast_factor + mean
            else:
                mean = image.mean()
                image = (image - mean) * contrast_factor + mean
            
        return image, mask 