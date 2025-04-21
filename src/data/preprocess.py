import os
import numpy as np
import argparse
import glob
from PIL import Image
import multiprocessing
from tqdm import tqdm
import skimage.transform as transform


def skull_strip(image, threshold=0.1):
    """
    Simple threshold-based skull stripping.
    
    Args:
        image (np.ndarray): Input image
        threshold (float): Threshold value (relative to max intensity)
    
    Returns:
        np.ndarray: Skull-stripped image
    """
    # Create a binary mask (brain vs. non-brain)
    mask = image > (threshold * np.max(image))
    
    # Apply the mask to the original image
    stripped = image * mask
    
    return stripped


def normalize_image(image, method='zscore'):
    """
    Normalize image intensities.
    
    Args:
        image (np.ndarray): Input image
        method (str): Normalization method ('zscore', 'minmax', or 'none')
    
    Returns:
        np.ndarray: Normalized image
    """
    if method == 'none':
        return image
    
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image
    
    if method == 'zscore':
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        return image
    
    raise ValueError(f"Unknown normalization method: {method}")


def resize_image(image, target_size=(256, 256), is_mask=False):
    """
    Resize image to target size.
    
    Args:
        image (np.ndarray): Input image
        target_size (tuple): Target size (height, width)
        is_mask (bool): Whether the image is a binary mask
    
    Returns:
        np.ndarray: Resized image
    """
    # Use nearest neighbor for masks to preserve binary values
    order = 0 if is_mask else 1
    
    # Resize using scikit-image
    return transform.resize(image, target_size, order=order, preserve_range=True)


def process_slice(args):
    """
    Process a single MRI slice and its corresponding mask.
    
    Args:
        args (tuple): (slice_path, mask_path, output_dir, params)
    
    Returns:
        str: Path to saved processed file
    """
    slice_path, mask_path, output_dir, params = args
    
    # Extract patient_id and slice_id from the path
    filename = os.path.basename(slice_path)
    patient_id = os.path.basename(os.path.dirname(slice_path))
    slice_id = os.path.splitext(filename)[0]
    
    # Create output directory
    patient_output_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # Load slice and mask
    slice_img = np.array(Image.open(slice_path))
    mask_img = np.array(Image.open(mask_path)) if mask_path else None
    
    # Resize if needed
    if params['resize']:
        slice_img = resize_image(slice_img, params['target_size'])
        if mask_img is not None:
            mask_img = resize_image(mask_img, params['target_size'], is_mask=True)
    
    # Apply skull stripping if requested
    if params['skull_strip']:
        slice_img = skull_strip(slice_img, params['threshold'])
    
    # Apply normalization
    slice_img = normalize_image(slice_img, params['normalization'])
    
    # Create dictionary to save
    data = {
        'image': slice_img,
        'mask': mask_img if mask_img is not None else np.zeros_like(slice_img),
        'patient_id': patient_id,
        'slice_id': slice_id,
    }
    
    # Save as numpy file
    output_path = os.path.join(patient_output_dir, f"{slice_id}.npy")
    np.save(output_path, data)
    
    return output_path


def preprocess_dataset(input_dir, output_dir, mask_dir=None, workers=4, **kwargs):
    """
    Preprocess entire dataset of MRI slices.
    
    Args:
        input_dir (str): Directory containing raw MRI slices
        output_dir (str): Directory to save processed files
        mask_dir (str): Directory containing mask files
        workers (int): Number of parallel workers
        **kwargs: Additional parameters for processing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image slices
    slice_paths = glob.glob(os.path.join(input_dir, '**', '*.tif'), recursive=True)
    print(f"Found {len(slice_paths)} slices")
    
    # Default parameters
    params = {
        'skull_strip': kwargs.get('skull_strip', False),
        'threshold': kwargs.get('threshold', 0.1),
        'normalization': kwargs.get('normalization', 'zscore'),
        'resize': kwargs.get('resize', False),
        'target_size': kwargs.get('target_size', (256, 256)),
    }
    
    # Create arguments for parallel processing
    args = []
    for slice_path in slice_paths:
        # If mask_dir is provided, try to find the corresponding mask
        mask_path = None
        if mask_dir:
            rel_path = os.path.relpath(slice_path, input_dir)
            mask_path = os.path.join(mask_dir, rel_path)
            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for {slice_path}")
                mask_path = None
        
        args.append((slice_path, mask_path, output_dir, params))
    
    # Process slices in parallel
    with multiprocessing.Pool(workers) as pool:
        processed_paths = list(tqdm(pool.imap(process_slice, args), total=len(args)))
    
    print(f"Processed {len(processed_paths)} slices")
    print(f"Saved to {output_dir}")


def cli():
    """Command-line interface for preprocessing data."""
    parser = argparse.ArgumentParser(description="Preprocess MRI data")
    parser.add_argument("--input_dir", required=True, help="Directory with raw MRI images")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed files")
    parser.add_argument("--mask_dir", help="Directory with mask images")
    parser.add_argument("--skull_strip", action="store_true", help="Apply skull stripping")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for skull stripping")
    parser.add_argument("--normalization", choices=["zscore", "minmax", "none"], default="zscore",
                        help="Intensity normalization method")
    parser.add_argument("--resize", action="store_true", help="Resize images")
    parser.add_argument("--width", type=int, default=256, help="Target width for resizing")
    parser.add_argument("--height", type=int, default=256, help="Target height for resizing")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        workers=args.workers,
        skull_strip=args.skull_strip,
        threshold=args.threshold,
        normalization=args.normalization,
        resize=args.resize,
        target_size=(args.height, args.width),
    )


if __name__ == "__main__":
    cli() 