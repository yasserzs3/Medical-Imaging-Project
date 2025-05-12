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
            angle = random.uniform(-self.degrees, self.degrees)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
        return image, mask


class RandomShift:
    """Randomly shift image and mask."""
    def __init__(self, shift_limit=0.05):
        self.shift_limit = shift_limit
        
    def __call__(self, image, mask):
        if random.random() < 0.5:
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
            
        return image, mask


class RandomZoom:
    """Randomly zoom image and mask."""
    def __init__(self, zoom_limit=0.05):
        self.zoom_limit = zoom_limit
        
    def __call__(self, image, mask):
        if random.random() < 0.5:
            height, width = image.shape[1:]
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
            
        return image, mask


class TransformCompose:
    """Compose multiple transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
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
        return TransformCompose([
            RandomFlip(p=0.5),
            RandomRotation(degrees=20),  
            RandomShift(shift_limit=0.05),
            RandomZoom(zoom_limit=0.05),
        ])
    else:
        return TransformCompose([])  # No augmentations for val/test 