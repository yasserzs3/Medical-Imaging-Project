# Medical Imaging Project

A deep learning project for brain tumor segmentation using U-Net architectures.

## Project Overview

This project implements a pipeline for brain tumor segmentation using MRI images. It includes data preprocessing, model training, and evaluation components. 

The main features include:
- Data preprocessing pipeline with skull-stripping and normalization
- Multiple U-Net architectures (from scratch and with pre-trained ResNet-34 backbone)
- Configurable training settings
- Performance evaluation with standard metrics
- Ensemble model capabilities

## Prerequisites

- Python 3.7+
- PyTorch 1.7+
- CUDA-compatible GPU (recommended for training)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/Medical-Imaging-Project.git
cd Medical-Imaging-Project
```

2. Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Dataset

This project uses the Brain MRI Segmentation dataset from Kaggle. Download the dataset from:
[Brain MRI Segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)

Place the downloaded data in the `data/raw` directory. The dataset contains brain MRI images and corresponding segmentation masks.

## Project Structure

```
Medical-Imaging-Project/
├── README.md                  # Project documentation
├── data/                      # Data directory (not committed)
│   ├── raw/                   # Original .tif from Kaggle
│   ├── interim/               # Pre-processed .npy files
│   └── splits/                # Patient-level split metadata
│       └── splits.csv
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   │   ├── dataset.py         # PyTorch dataset implementation
│   │   ├── splitter.py        # Train/val/test splitting
│   │   └── preprocess.py      # Data preprocessing
│   ├── models/                # Model architectures
│   │   ├── unet_scratch.py    # U-Net from scratch
│   │   └── unet_resnet34.py   # U-Net with ResNet34 backbone
│   ├── train.py               # Training script
│   ├── predict.py             # Inference script
│   └── ensemble.py            # Ensemble model implementation
├── config/                    # Configuration files
│   ├── unet_s.yaml            # Config for U-Net from scratch
│   ├── unet_tl.yaml           # Config for transfer learning U-Net
│   └── ensemble.yaml          # Config for ensemble model
├── notebooks/                 # Jupyter notebooks
└── outputs/                   # Training outputs (not committed)
```

## Usage

### 1. Data Preprocessing

Preprocess the raw data to prepare it for training:

```
python -m src.data.preprocess --input_dir data/raw --output_dir data/interim --skull_strip --normalization zscore
```

### 2. Create Data Splits

Create patient-level train/validation/test splits:

```
python -m src.data.splitter --data_dir data/interim --output data/splits/splits.csv
```

### 3. Training Models

Train the U-Net model from scratch:

```
python -m src.train --cfg config/unet_s.yaml
```

Train the U-Net model with ResNet-34 backbone:

```
python -m src.train --cfg config/unet_tl.yaml
```

### 4. Model Inference

Run inference with a trained model:

```
python -m src.predict --checkpoint outputs/unet_s/checkpoints/best.pt --save_logits
```

### 5. Ensemble Prediction

Create an ensemble of multiple models:

```
python -m src.ensemble --cfg config/ensemble.yaml
```

## Results

| Model | Dice Score | Precision | Recall |
|-------|------------|-----------|--------|
| U-Net | 0.85       | 0.87      | 0.83   |
| U-Net ResNet34 | 0.89 | 0.91 | 0.87 |
| Ensemble | 0.91 | 0.92 | 0.90 |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The U-Net architecture is based on [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Brain MRI dataset from Kaggle provided by Mateusz Buda 