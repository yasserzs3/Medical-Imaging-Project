import os
import argparse
import numpy as np
import glob
from tqdm import tqdm
import yaml
from PIL import Image
import matplotlib.pyplot as plt


def load_logits(logits_paths):
    """
    Load multiple logits files.
    
    Args:
        logits_paths (list): List of paths to logits files
    
    Returns:
        list: List of logits arrays
    """
    logits_list = []
    for path in logits_paths:
        logits = np.load(path)
        logits_list.append(logits)
    
    return logits_list


def soft_vote(logits_list, weights=None):
    """
    Perform soft voting ensemble of multiple logits.
    
    Args:
        logits_list (list): List of logits arrays
        weights (list): Optional weights for each model
    
    Returns:
        np.ndarray: Ensemble logits
    """
    if weights is None:
        weights = [1.0] * len(logits_list)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Weighted average of logits
    ensemble_logits = np.zeros_like(logits_list[0])
    for i, logits in enumerate(logits_list):
        ensemble_logits += weights[i] * logits
    
    return ensemble_logits


def save_ensemble_prediction(original_image, ensemble_probs, threshold, output_path):
    """
    Save ensemble prediction visualization.
    
    Args:
        original_image (np.ndarray): Original image
        ensemble_probs (np.ndarray): Ensemble probability map
        threshold (float): Threshold for binary prediction
        output_path (str): Path to save visualization
    """
    # Create binary prediction
    pred_mask = (ensemble_probs > threshold).astype(np.uint8)
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(pred_mask, cmap='gray')
    plt.title(f'Ensemble Prediction (t={threshold})')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(ensemble_probs, cmap='jet')
    plt.colorbar()
    plt.title('Ensemble Probability Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_ensemble(config_path, output_dir=None):
    """
    Create an ensemble from multiple model predictions.
    
    Args:
        config_path (str): Path to ensemble configuration file
        output_dir (str): Directory to save ensemble predictions
    """
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join('outputs', 'ensemble', os.path.splitext(os.path.basename(config_path))[0])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model predictions
    models = cfg['models']
    weights = cfg.get('weights', [1.0] * len(models))
    threshold = cfg.get('threshold', 0.5)
    
    print(f"Creating ensemble of {len(models)} models with threshold {threshold}")
    
    # Process all prediction files for each patient/slice
    for split in ['val', 'test']:
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Get all patient directories from the first model
        first_model_dir = models[0]['predictions_dir']
        patient_dirs = glob.glob(os.path.join(first_model_dir, split, '*'))
        
        for patient_dir in patient_dirs:
            patient_id = os.path.basename(patient_dir)
            patient_output_dir = os.path.join(split_output_dir, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)
            
            # Get all slice logits files
            slice_files = glob.glob(os.path.join(patient_dir, '*_logits.npy'))
            
            for slice_file in tqdm(slice_files, desc=f"Processing {patient_id} ({split})"):
                slice_id = os.path.basename(slice_file).replace('_logits.npy', '')
                
                # Collect logits from all models
                logits_paths = []
                for model in models:
                    model_slice_path = os.path.join(
                        model['predictions_dir'], 
                        split, 
                        patient_id, 
                        f"{slice_id}_logits.npy"
                    )
                    if os.path.exists(model_slice_path):
                        logits_paths.append(model_slice_path)
                
                if len(logits_paths) < len(models):
                    print(f"Warning: Some models don't have predictions for {slice_id}")
                    continue
                
                # Load logits and create ensemble
                logits_list = load_logits(logits_paths)
                ensemble_probs = soft_vote(logits_list, weights[:len(logits_list)])
                
                # Save ensemble probabilities
                ensemble_logits_path = os.path.join(patient_output_dir, f"{slice_id}_ensemble_logits.npy")
                np.save(ensemble_logits_path, ensemble_probs)
                
                # Get original image for visualization
                # Try to find the original image
                original_data_path = None
                for model in models:
                    # Look for the original data path in model predictions
                    if 'data_dir' in model:
                        potential_path = os.path.join(model['data_dir'], patient_id, f"{slice_id}.npy")
                        if os.path.exists(potential_path):
                            original_data_path = potential_path
                            break
                
                if original_data_path:
                    # Load original image
                    original_data = np.load(original_data_path, allow_pickle=True).item()
                    original_image = original_data['image']
                    
                    # Save visualization
                    visualization_path = os.path.join(patient_output_dir, f"{slice_id}_ensemble.png")
                    save_ensemble_prediction(original_image, ensemble_probs, threshold, visualization_path)
                else:
                    print(f"Warning: Could not find original image for {slice_id}")
    
    print(f"Ensemble predictions saved to {output_dir}")


def cli():
    """Command-line interface for creating ensembles."""
    parser = argparse.ArgumentParser(description="Create ensemble predictions")
    parser.add_argument("--cfg", required=True, help="Path to ensemble config YAML file")
    parser.add_argument("--output_dir", help="Directory to save ensemble predictions")
    
    args = parser.parse_args()
    create_ensemble(args.cfg, args.output_dir)


if __name__ == "__main__":
    cli() 