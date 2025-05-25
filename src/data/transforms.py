"""Data augmentation transformations for medical image segmentation."""

import random
import torch
import torchvision.transforms as transforms
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

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


# New augmentation classes

class RandomIntensity:
    """Randomly adjust brightness and contrast of the image."""
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        
    def __call__(self, image, mask):
        if random.random() < 0.5:
            # Random brightness
            brightness_factor = random.uniform(*self.brightness_range)
            image = image * brightness_factor
            image = torch.clamp(image, 0, 1)
            
        if random.random() < 0.5:
            # Random contrast
            contrast_factor = random.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * contrast_factor + mean
            image = torch.clamp(image, 0, 1)
            
        return image, mask


class RandomNoise:
    """Add random Gaussian noise to the image."""
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level
        
    def __call__(self, image, mask):
        if random.random() < 0.5:
            noise = torch.randn_like(image) * self.noise_level
            image = image + noise
            image = torch.clamp(image, 0, 1)
        return image, mask


class RandomBlur:
    """Apply random Gaussian blur to the image."""
    def __init__(self, kernel_range=(3, 7), sigma_range=(0.1, 2.0)):
        self.kernel_range = kernel_range
        self.sigma_range = sigma_range
        
    def __call__(self, image, mask):
        if random.random() < 0.3:  # Lower probability as it's a strong augmentation
            # Convert to numpy for gaussian blur
            image_np = image.squeeze().numpy()
            
            # Apply gaussian blur
            kernel_size = random.randrange(*self.kernel_range)
            if kernel_size % 2 == 0:  # Ensure kernel size is odd
                kernel_size += 1
                
            sigma = random.uniform(*self.sigma_range)
            blurred = gaussian_filter(image_np, sigma=sigma)
            
            # Convert back to torch
            image = torch.from_numpy(blurred).unsqueeze(0) if image.shape[0] == 1 else torch.from_numpy(blurred)
            
        return image, mask


class RandomElasticDeformation:
    """Apply elastic deformation to image and mask."""
    def __init__(self, alpha=1000, sigma=20):
        self.alpha = alpha
        self.sigma = sigma
        
    def __call__(self, image, mask):
        if random.random() < 0.3:  # Lower probability as it's a strong augmentation
            # Store original shapes and devices
            device = image.device
            image_shape = image.shape
            mask_shape = mask.shape
            
            # Convert to numpy
            image_np = image.cpu().squeeze().numpy()
            mask_np = mask.cpu().squeeze().numpy()
            
            shape = image_np.shape
            
            # Generate random displacement fields
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            
            # Create meshgrid
            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            
            # Apply displacement
            indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
            
            # Apply to image and mask
            image_deformed = map_coordinates(image_np, indices, order=1).reshape(shape)
            mask_deformed = map_coordinates(mask_np, indices, order=0).reshape(shape)
            
            # Convert back to torch and reshape
            image = torch.from_numpy(image_deformed).float()
            mask = torch.from_numpy(mask_deformed).float()
            
            # Restore original shape
            if len(image_shape) > 2:
                image = image.unsqueeze(0)
            if len(mask_shape) > 2:
                mask = mask.unsqueeze(0)
                
            # Move back to original device
            image = image.to(device)
            mask = mask.to(device)
            
        return image, mask


class RandomGamma:
    """Apply random gamma correction to the image."""
    def __init__(self, gamma_range=(0.7, 1.5)):
        self.gamma_range = gamma_range
        
    def __call__(self, image, mask):
        if random.random() < 0.5:
            gamma = random.uniform(*self.gamma_range)
            image = torch.pow(image, gamma)
        return image, mask


class RandomCutout:
    """Randomly cut out rectangular regions from the image and mask."""
    def __init__(self, max_holes=3, max_height=20, max_width=20):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        
    def __call__(self, image, mask):
        if random.random() < 0.3:  # Lower probability
            h, w = image.shape[1:]
            
            # Number of holes to cut out
            n_holes = random.randint(1, self.max_holes)
            
            for _ in range(n_holes):
                # Random position and size
                y = random.randint(0, h - 1)
                x = random.randint(0, w - 1)
                
                y1 = max(0, y - random.randint(1, self.max_height // 2))
                y2 = min(h, y + random.randint(1, self.max_height // 2))
                x1 = max(0, x - random.randint(1, self.max_width // 2))
                x2 = min(w, x + random.randint(1, self.max_width // 2))
                
                # Cut out the region
                image[:, y1:y2, x1:x2] = 0
                
                # We don't modify the mask as it would affect ground truth
                
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
            # Spatial transforms
            RandomFlip(p=0.5),
            RandomRotation(degrees=30),  # Increased from 20
            RandomShift(shift_limit=0.05),
            RandomZoom(zoom_limit=0.1),  # Increased from 0.05
            
            # Intensity transforms
            RandomIntensity(brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)),
            RandomGamma(gamma_range=(0.7, 1.5)),
            RandomNoise(noise_level=0.05),
            RandomBlur(kernel_range=(3, 7), sigma_range=(0.1, 2.0)),
            
            # Advanced transforms
            RandomElasticDeformation(alpha=1000, sigma=20),
            RandomCutout(max_holes=2, max_height=20, max_width=20),
        ]
        # Wrap with consistency checker
        return EnsureSizeConsistency(TransformCompose(transforms_list))
    else:
        return TransformCompose([])  # No augmentations for val/test 