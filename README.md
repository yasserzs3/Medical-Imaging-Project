# Brain MRI Segmentation Project

A deep learning project for segmenting brain MRI scans using a UNet architecture.

## Project Structure

```
.
├── data/
│   └── raw/                 # Raw MRI images and masks
├── outputs/                 # Training outputs and saved models
├── src/
│   ├── data/                # Data loading and preprocessing
│   │   ├── dataset.py       # Dataset classes and data loading utilities
│   │   └── transforms.py    # Data augmentation transformations
│   └── models/              # Model architectures
│       ├── losses.py        # Loss functions 
│       ├── metrics.py       # Evaluation metrics
│       └── unet.py          # UNet implementation
└── train_lgg_mri.py         # Main training script
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare your data in the `data/raw` directory with the following structure:
   - Each patient should have a folder
   - Images and masks should be paired, with masks having "_mask" suffix

## Training

To train the model with default parameters:

```bash
python train_lgg_mri.py
```

For additional options:

```bash
python train_lgg_mri.py --help
```

Common parameters:
- `--data_dir`: Path to data directory (default: "data/raw")
- `--batch_size`: Batch size (default: 16)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 0.0001)
- `--in_channels`: Number of input channels (1 for grayscale, 3 for RGB)
- `--augment`: Use data augmentation
- `--loss_type`: Loss function to use (choices: bce, dice, iou, bce_dice, bce_iou, combo)

## Results

Training results are saved in the `outputs` directory with timestamped folders containing:
- Model checkpoints
- TensorBoard logs
- Training arguments and results 