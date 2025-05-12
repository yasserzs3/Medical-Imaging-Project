"""Data augmentation transformations for medical image segmentation."""

import random
import torch
import torchvision.transforms as transforms

class RandomFlip:
    """Randomly flip image and mask horizontally."""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = torch.flip(image, dims=[2])  # Flip horizontally
            mask = torch.flip(mask, dims=[2])
        return image, mask


class RandomRotation:
    """Randomly rotate image and mask."""
    def __init__(self, degrees=20):
        self.degrees = degrees
        
    def __call__(self, image, mask):
        if random.random() < 0.5:
            # Store original size
            orig_size = image.shape[-2:]
            
            # Perform rotation
            angle = random.uniform(-self.degrees, self.degrees)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
            
            # Resize back to original dimensions to ensure consistent sizes
            if image.shape[-2:] != orig_size:
                image = transforms.functional.resize(image, orig_size, 
                                               transforms.InterpolationMode.BILINEAR)
                mask = transforms.functional.resize(mask, orig_size, 
                                              transforms.InterpolationMode.NEAREST)
        
        return image, mask


class RandomShift:
    """Randomly shift image and mask."""
    def __init__(self, shift_limit=0.05):
        self.shift_limit = shift_limit
        
    def __call__(self, image, mask):
        if random.random() < 0.5:
            # Store original size
            orig_size = image.shape[-2:]
            
            height, width = image.shape[1:]
            
            # Calculate shift amount
            shift_x = random.uniform(-self.shift_limit, self.shift_limit) * width
            shift_y = random.uniform(-self.shift_limit, self.shift_limit) * height
            
            # Convert to integers
            shift_x = int(shift_x)
            shift_y = int(shift_y)
            
            # Create the translation matrix
            translate = (shift_x, shift_y)
            
            # Apply the translation
            image = transforms.functional.affine(image, angle=0, translate=translate, scale=1.0, shear=0)
            mask = transforms.functional.affine(mask, angle=0, translate=translate, scale=1.0, shear=0)
            
            # Resize back to original dimensions if needed
            if image.shape[-2:] != orig_size:
                image = transforms.functional.resize(image, orig_size, 
                                             transforms.InterpolationMode.BILINEAR)
                mask = transforms.functional.resize(mask, orig_size, 
                                             transforms.InterpolationMode.NEAREST)
            
        return image, mask


class RandomZoom:
    """Randomly zoom image and mask."""
    def __init__(self, zoom_limit=0.05):
        self.zoom_limit = zoom_limit
        
    def __call__(self, image, mask):
        if random.random() < 0.5:
            # Store original size
            orig_size = image.shape[-2:]
            
            # Calculate random scale factor
            scale = random.uniform(1 - self.zoom_limit, 1 + self.zoom_limit)
            
            # Apply zoom using scale factor
            image = transforms.functional.affine(
                image, 
                angle=0, 
                translate=(0, 0), 
                scale=scale, 
                shear=0,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
            
            mask = transforms.functional.affine(
                mask, 
                angle=0, 
                translate=(0, 0), 
                scale=scale, 
                shear=0,
                interpolation=transforms.InterpolationMode.NEAREST
            )
            
            # Resize back to original dimensions
            if image.shape[-2:] != orig_size:
                image = transforms.functional.resize(
                    image, 
                    orig_size, 
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
                
                mask = transforms.functional.resize(
                    mask, 
                    orig_size, 
                    interpolation=transforms.InterpolationMode.NEAREST
                )
                
        return image, mask


class TransformCompose:
    """Compose multiple transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class EnsureSizeConsistency:
    """Ensure that images and masks maintain consistent size throughout transformations."""
    def __init__(self, transform, size=None):
        """
        Args:
            transform: The transform pipeline to apply
            size: The target size (H, W) to enforce, or None to use input size
        """
        self.transform = transform
        self.size = size
        
    def __call__(self, image, mask):
        # Store original size if no specific size is provided
        if self.size is None:
            target_size = image.shape[-2:]
        else:
            target_size = self.size
            
        # Apply transformations
        image, mask = self.transform(image, mask)
        
        # Resize to original dimensions if needed
        if image.shape[-2:] != target_size:
            image = transforms.functional.resize(
                image, 
                target_size, 
                interpolation=transforms.InterpolationMode.BILINEAR
            )
            
            mask = transforms.functional.resize(
                mask, 
                target_size, 
                interpolation=transforms.InterpolationMode.NEAREST
            )
            
        # Check channel dimension - ensure consistency
        if image.shape[0] != mask.shape[0]:
            # Typically we want to keep mask as 1-channel
            if mask.shape[0] > 1:
                # Convert multi-channel mask to single channel
                mask = mask[0].unsqueeze(0)
                
        return image, mask


def get_transforms(mode='train'):
    """
    Get transforms for training or validation/testing.
    
    Args:
        mode (str): 'train' for training transforms, 'val' for validation/testing
    
    Returns:
        TransformCompose: Composed transforms
    """
    if mode == 'train':
        transforms_list = [
            RandomFlip(p=0.5),
            RandomRotation(degrees=20),  
            RandomShift(shift_limit=0.05),
            RandomZoom(zoom_limit=0.05),
        ]
        # Wrap with consistency checker
        return EnsureSizeConsistency(TransformCompose(transforms_list))
    else:
        return TransformCompose([])  # No augmentations for val/test 