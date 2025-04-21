import os
import pandas as pd
import numpy as np
import argparse


def make_splits(data_dir, output_csv, seed=42, ratios=(0.7, 0.15, 0.15)):
    """
    Create patient-level train/val/test splits for medical imaging data.
    
    Args:
        data_dir (str): Directory containing patient data
        output_csv (str): Path to save split CSV file
        seed (int): Random seed for reproducibility
        ratios (tuple): Proportions for train, val, test splits (must sum to 1)
    
    Returns:
        pd.DataFrame: DataFrame with patient_id and split columns
    """
    # Validate ratios
    assert sum(ratios) == 1.0, "Split ratios must sum to 1"
    assert len(ratios) == 3, "Must provide 3 split ratios (train, val, test)"
    
    # Get all unique patient IDs
    patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Shuffle patient IDs
    np.random.shuffle(patient_dirs)
    
    # Calculate split indices
    n_patients = len(patient_dirs)
    n_train = int(n_patients * ratios[0])
    n_val = int(n_patients * ratios[1])
    
    # Create splits
    train_patients = patient_dirs[:n_train]
    val_patients = patient_dirs[n_train:n_train+n_val]
    test_patients = patient_dirs[n_train+n_val:]
    
    # Create DataFrame
    data = []
    for patient_id in train_patients:
        data.append({'patient_id': patient_id, 'split': 'train'})
    for patient_id in val_patients:
        data.append({'patient_id': patient_id, 'split': 'val'})
    for patient_id in test_patients:
        data.append({'patient_id': patient_id, 'split': 'test'})
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"Created splits with {len(train_patients)} train, {len(val_patients)} validation, "
          f"and {len(test_patients)} test patients")
    print(f"Saved to {output_csv}")
    
    return df


def cli():
    """Command-line interface for creating dataset splits."""
    parser = argparse.ArgumentParser(description="Create patient-level train/val/test splits")
    parser.add_argument("--data_dir", required=True, help="Directory containing patient data")
    parser.add_argument("--output", required=True, help="Path to save split CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion for training set")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proportion for validation set")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proportion for test set")
    
    args = parser.parse_args()
    
    # Verify that ratios sum to 1
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if sum(ratios) != 1.0:
        parser.error("Split ratios must sum to 1")
    
    make_splits(args.data_dir, args.output, args.seed, ratios)


if __name__ == "__main__":
    cli() 