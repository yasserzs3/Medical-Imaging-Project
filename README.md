# Brain MRI Segmentation Project

A deep learning project for segmenting brain MRI scans using a UNet architecture with comprehensive evaluation metrics and visualization tools.

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
│   ├── utlis/               # Utility scripts (Note: directory name as is)
│   │   ├── visualize_scans_masks.py  # Visualization utilities for scans and masks
│   │   └── inference_best_model.py   # Model inference utilities
│   ├── models/              # Model architectures
│   │   ├── unet.py          # UNet implementation
│   │   ├── losses.py        # Loss functions (BCE, Dice, IoU, Combo)
│   │   └── metrics.py       # Evaluation metrics and confusion matrix
│   └── visualization_results/ # Visualization outputs (ignored by git)
├── train_lgg_mri.py         # Main training script
├── metrics_plots.ipynb      # Jupyter notebook for plotting training metrics
└── requirements.txt         # Project dependencies
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

### Training Parameters

- `--data_dir`: Path to data directory (default: "data/raw/lgg-mri-segmentation/kaggle_3m")
- `--output_dir`: Directory to save models and logs (default: "outputs")
- `--batch_size`: Batch size (default: 16)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 0.0001)
- `--weight_decay`: Weight decay for optimizer (default: 0.00005)
- `--patience`: Patience for early stopping (default: 15)
- `--seed`: Random seed for reproducibility (default: 42)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--in_channels`: Number of input channels (1 for grayscale, 3 for RGB, default: 1)
- `--augment`: Use data augmentation (flag)
- `--no_cuda`: Disable CUDA (flag)
- `--loss_type`: Loss function to use (choices: "bce", "dice", "iou", "bce_dice", "bce_iou", "combo", default: "combo")

### Loss Functions

The project supports multiple loss functions:
- **BCE**: Binary Cross Entropy
- **Dice**: Dice Loss
- **IoU**: Intersection over Union Loss
- **BCE + Dice**: Combination of BCE and Dice (50/50)
- **BCE + IoU**: Combination of BCE and IoU (50/50)
- **Combo**: Advanced combination loss (default)

## Results and Monitoring

Training results are saved in the `outputs` directory with timestamped folders containing:
- **Model checkpoints**: `best_model.pt`, `final_model.pt`, and epoch checkpoints
- **TensorBoard logs**: Real-time training monitoring in `logs/` subdirectory
- **Training metrics**: `training_metrics.json` with complete training history
- **Final results**: `results.json` with best and final performance metrics
- **Training arguments**: `args.json` with all training parameters used

### Metrics Tracked

The training process tracks comprehensive metrics:
- Loss (training and validation)
- Dice Coefficient
- IoU (Intersection over Union)
- Accuracy, Precision, Recall, Specificity
- F1 Score
- Learning Rate

### Visualization

- **TensorBoard**: Use `tensorboard --logdir outputs/[timestamp]/logs` to monitor training
- **Metrics Plotting**: Use `metrics_plots.ipynb` to generate detailed plots from training results
- **Visualization results**: Stored in `src/visualization_results/` (ignored by git)

## Model Architecture

The project uses a UNet architecture specifically designed for medical image segmentation, with:
- Encoder-decoder structure with skip connections
- Configurable input channels (grayscale or RGB)
- Single output channel for binary segmentation
- Early stopping based on combined Dice + IoU metric

## Dependencies

- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- numpy >= 1.19.0
- pandas >= 1.1.0
- tqdm >= 4.48.0
- Pillow >= 7.2.0
- scikit-image >= 0.17.0
- opencv-python >= 4.4.0
- tensorboard >= 2.3.0 