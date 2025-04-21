import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
import random
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

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
            transform (callable, optional): Optional transform to apply to both image and mask
            augment (bool): Whether to apply augmentation (for training)
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
        
        # Load image and mask
        try:
            image = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
        except Exception as e:
            print(f"Error loading {img_path} or {mask_path}: {e}")
            # Return a dummy sample on error
            return torch.zeros((self.in_channels, 256, 256)), torch.zeros((1, 256, 256))
        
        # Debug info for the first sample
        if idx == 0:
            print(f"Original image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
            print(f"Original mask shape: {mask.shape}, dtype: {mask.dtype}, range: [{mask.min()}, {mask.max()}]")
        
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


# Enhanced data augmentation functions
class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = torch.flip(image, [2])  # Horizontal flip
            mask = torch.flip(mask, [2])
        return image, mask

class RandomRotation:
    def __init__(self, degrees=20):
        self.degrees = degrees
        
    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        return image, mask

class RandomShift:
    def __init__(self, shift_limit=0.05):
        self.shift_limit = shift_limit
        
    def __call__(self, image, mask):
        height, width = image.shape[1:]
        
        # Random shifts
        h_shift = int(height * random.uniform(-self.shift_limit, self.shift_limit))
        w_shift = int(width * random.uniform(-self.shift_limit, self.shift_limit))
        
        # Apply shifts using padding and cropping
        if h_shift > 0:
            image = torch.cat([torch.zeros_like(image[:, :h_shift, :]), image[:, :-h_shift, :]], dim=1)
            mask = torch.cat([torch.zeros_like(mask[:, :h_shift, :]), mask[:, :-h_shift, :]], dim=1)
        elif h_shift < 0:
            image = torch.cat([image[:, -h_shift:, :], torch.zeros_like(image[:, :h_shift, :])], dim=1)
            mask = torch.cat([mask[:, -h_shift:, :], torch.zeros_like(mask[:, :h_shift, :])], dim=1)
            
        if w_shift > 0:
            image = torch.cat([torch.zeros_like(image[:, :, :w_shift]), image[:, :, :-w_shift]], dim=2)
            mask = torch.cat([torch.zeros_like(mask[:, :, :w_shift]), mask[:, :, :-w_shift]], dim=2)
        elif w_shift < 0:
            image = torch.cat([image[:, :, -w_shift:], torch.zeros_like(image[:, :, :w_shift])], dim=2)
            mask = torch.cat([mask[:, :, -w_shift:], torch.zeros_like(mask[:, :, :w_shift])], dim=2)
            
        return image, mask

class RandomZoom:
    def __init__(self, zoom_limit=0.05):
        self.zoom_limit = zoom_limit
        
    def __call__(self, image, mask):
        height, width = image.shape[1:]
        
        # Random scale factor
        scale = random.uniform(1 - self.zoom_limit, 1 + self.zoom_limit)
        
        if scale != 1:
            # Resize using torchvision
            new_h = int(height * scale)
            new_w = int(width * scale)
            
            # Resize with bilinear interpolation for image
            image = transforms.functional.resize(image, (new_h, new_w), 
                                               transforms.InterpolationMode.BILINEAR)
            
            # Resize with nearest neighbor for mask to preserve binary values
            mask = transforms.functional.resize(mask, (new_h, new_w), 
                                              transforms.InterpolationMode.NEAREST)
            
            # Crop or pad to original size
            if new_h > height:
                diff = new_h - height
                image = image[:, diff//2:diff//2 + height, :]
                mask = mask[:, diff//2:diff//2 + height, :]
            elif new_h < height:
                diff = height - new_h
                padding = (0, 0, 0, 0, diff//2, diff - diff//2)
                image = torch.nn.functional.pad(image, padding)
                mask = torch.nn.functional.pad(mask, padding)
                
            if new_w > width:
                diff = new_w - width
                image = image[:, :, diff//2:diff//2 + width]
                mask = mask[:, :, diff//2:diff//2 + width]
            elif new_w < width:
                diff = width - new_w
                padding = (diff//2, diff - diff//2, 0, 0, 0, 0)
                image = torch.nn.functional.pad(image, padding)
                mask = torch.nn.functional.pad(mask, padding)
                
        return image, mask

class RandomShear:
    def __init__(self, shear_limit=0.05):
        self.shear_limit = shear_limit
        
    def __call__(self, image, mask):
        shear = random.uniform(-self.shear_limit, self.shear_limit)
        
        # Apply shear using affine transform
        angle = 0.0
        translate = (0.0, 0.0)
        scale = 1.0
        
        image = transforms.functional.affine(image, angle, translate, scale, [shear, 0.0])
        mask = transforms.functional.affine(mask, angle, translate, scale, [shear, 0.0])
        
        return image, mask

class RandomBrightnessContrast:
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast
        
    def __call__(self, image, mask):
        if random.random() < 0.5:
            factor = random.uniform(1-self.brightness, 1+self.brightness)
            image = transforms.functional.adjust_brightness(image, factor)
        
        if random.random() < 0.5:
            factor = random.uniform(1-self.contrast, 1+self.contrast)
            image = transforms.functional.adjust_contrast(image, factor)
            
        return image, mask

class ElasticTransform:
    """
    Apply elastic transformation to images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
    """
    def __init__(self, alpha=1, sigma=10, alpha_affine=10, p=0.5):
        """
        Args:
            alpha: Intensity of deformation (larger values mean more deformation)
            sigma: Gaussian filter parameter controlling elasticity
            alpha_affine: Range of affine degrees
            p: Probability of applying the transform
        """
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.p = p
        
    def _elastic_transform(self, image, mask):
        """Apply elastic transform to image and mask."""
        # Convert PyTorch tensors to NumPy for this transform
        if isinstance(image, torch.Tensor):
            was_tensor = True
            image_np = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            mask_np = mask.permute(1, 2, 0).cpu().numpy()
        else:
            was_tensor = False
            image_np = image
            mask_np = mask
            
        shape = image_np.shape[:2]  # (height, width)
        
        # Random affine
        if random.random() < 0.5:
            center_square = np.float32(shape) // 2
            square_size = min(shape) // 3
            pts1 = np.float32([
                center_square + square_size,
                [center_square[0]+square_size, center_square[1]-square_size],
                center_square - square_size
            ])
            pts2 = pts1 + np.random.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
            M = cv2.getAffineTransform(pts1, pts2)
            image_np = cv2.warpAffine(image_np, M, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
            mask_np = cv2.warpAffine(mask_np, M, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
        
        # Elastic transform
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        # Apply displacement
        distorted_image = np.zeros_like(image_np)
        # Handle both grayscale (2D) and color (3D) images
        if len(image_np.shape) > 2 and image_np.shape[2] > 1:
            # Color image with multiple channels
            for c in range(image_np.shape[2]):
                distorted_image[..., c] = map_coordinates(image_np[..., c], indices, order=1, mode='reflect').reshape(shape)
        else:
            # Grayscale image (handle as single channel)
            if len(image_np.shape) == 2:
                # True 2D array
                distorted_image = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)
            else:
                # 3D array with single channel
                distorted_image[..., 0] = map_coordinates(image_np[..., 0], indices, order=1, mode='reflect').reshape(shape)
            
        distorted_mask = np.zeros_like(mask_np)
        # Handle both grayscale (2D) and color (3D) masks
        if len(mask_np.shape) > 2 and mask_np.shape[2] > 1:
            # Multi-channel mask
            for c in range(mask_np.shape[2]):
                distorted_mask[..., c] = map_coordinates(mask_np[..., c], indices, order=0, mode='reflect').reshape(shape)
        else:
            # Single channel mask
            if len(mask_np.shape) == 2:
                # True 2D array
                distorted_mask = map_coordinates(mask_np, indices, order=0, mode='reflect').reshape(shape)
            else:
                # 3D array with single channel
                distorted_mask[..., 0] = map_coordinates(mask_np[..., 0], indices, order=0, mode='reflect').reshape(shape)
        
        # Convert back to tensor if necessary
        if was_tensor:
            # Ensure arrays have proper shape with channel dimension
            if len(distorted_image.shape) == 2:
                distorted_image = distorted_image[..., np.newaxis]
            if len(distorted_mask.shape) == 2:
                distorted_mask = distorted_mask[..., np.newaxis]
                
            distorted_image = torch.from_numpy(distorted_image).permute(2, 0, 1)  # HWC -> CHW
            distorted_mask = torch.from_numpy(distorted_mask).permute(2, 0, 1)
            # Ensure mask is binary
            distorted_mask = (distorted_mask > 0.5).float()
        
        return distorted_image, distorted_mask
        
    def __call__(self, image, mask):
        if random.random() < self.p:
            return self._elastic_transform(image, mask)
        return image, mask

class TransformCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

def get_transforms(mode='train'):
    """Get transforms for training or validation/testing."""
    if mode == 'train':
        return TransformCompose([
            RandomFlip(p=0.5),
            RandomRotation(degrees=20),  # 0.2 radians ≈ 11.5 degrees
            RandomShift(shift_limit=0.05),
            RandomShear(shear_limit=0.05),
            RandomZoom(zoom_limit=0.05),
            ElasticTransform(alpha=1, sigma=10, p=0.2),  # Lower probability for elastic transform
            RandomBrightnessContrast(brightness=0.2, contrast=0.2),
        ])
    else:
        return TransformCompose([])  # No augmentations for val/test 