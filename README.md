## Deep Learning Approaches for Moth Species Classification and Conservation Monitoring
This repository contains the code, results, and visualisations for the MSc Data Science Extended Research Project at the University of Manchester investigating automated moth species classification using deep learning architectures.

### Project Overview
This study develops a systematic methodology using deep learning for automated moth species classification to identify optimal architectures and assess deployment suitability for conservation monitoring practices. Seven architectures including CNNs, Vision Transformers, and hybrid models were evaluated on 50 moth species using a four-phase optimisation pipeline.

### Repository Structure
moth-classification-research-project/
├── notebooks/                    
│   ├── 01_resnet50_baseline.ipynb           # Frozen ResNet50 control
│   ├── 02_resnet50_partial_ft.ipynb         # ResNet50 with partial fine-tuning
│   ├── 03_efficientnet_b0_partial_ft.ipynb  # EfficientNet_B0 baseline
│   ├── 04_inception_v3_partial_ft.ipynb     # InceptionV3 baseline
│   ├── 05_vgg16_partial_ft.ipynb            # VGG16 baseline
│   ├── 06_mobilenet_v2_partial_ft.ipynb     # MobileNetV2 baseline
│   ├── 07_vit_b16_partial_ft.ipynb          # ViT-B/16 baseline
│   ├── 08_coat_lite_medium_partial_ft.ipynb # CoaT-Lite Medium baseline
│   ├── 11_efficientnet_full_ft.ipynb        # EfficientNet_B0 optimisation pipeline
│   ├── 12_mobilenet_v2_full_ft.ipynb        # MobileNetV2 optimisation pipeline
│   ├── 13_vit_b16_full_ft.ipynb             # ViT-B/16 optimisation pipeline
│   ├── 14_coat_lite_medium_full_ft.ipynb    # CoaT-Lite optimisation pipeline
│   ├── confused-species-extra-transformations.ipynb  # Targeted augmentation application
│   ├── data-exploration-and-figures.ipynb   # Dataset analysis & thesis figure creation
│   └── *.pth files                          # Trained model weight details
├── figures/                      # Generated figures and plots
├── tables/                       # Results tables and summaries
├── cam_outputs/                  # Score-CAM and Grad-CAM++ visualisations
│   ├── efficientnet_b0/         # EfficientNet_B0 Baseline: Grad-CAM++ (CutMix stage)
│   ├── efficientnet_b0_ft/      # EfficientNet_B0 Fine-tuned: Grad-CAM++ (remaining stages)
│   ├── mobilenetv2/             # MobileNetV2 Baseline: Score-CAM (CutMix stage) 
│   ├── mobilenetv2_ft/          # MobileNetV2 Fine-tuned: Score-CAM (remaining stages)
│   ├── vit_b16/                 # ViT-B/16 Baseline: Score-CAM (CutMix stage)
│   ├── vit_b16_ft/              # ViT-B/16 Fine-tuned: Score-CAM (remaining stages)
│   ├── coat_lite/               # CoaT-Lite Medium Baseline: Score-CAM (CutMix stage)
│   └── coat_lite_ft/            # CoaT-Lite Medium Fine-tuned: Score-CAM (remaining stages)
├── data/                        # Dataset directory (not tracked)
│   └── kaggle-dataset/          # Moth images dataset
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── .gitignore                   # Git ignore rules
└── README.md                    # README file

### Methodological Contributions

- Four-phase optimisation pipeline: Baseline → CAM analysis → Targeted augmentation → Deep fine-tuning
- Interpretability-driven analysis: Using Grad-CAM++ and Score-CAM to guide augmentation strategies
- Systematic evaluation: Isolated effects of each optimisation technique

## Setup Instructions
### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- At least 16GB RAM

### Installation
#### 1. Clone the repository

git clone https://github.com/Reb-jon/moth-classification-research-project.git
cd moth-classification-research-project

## Datasets Setup

This project uses one dataset. Please download the dataset manually and place them in the `data/` folder.

### 1. Kaggle Moths Dataset
- Source: [Kaggle - Moths Image Dataset](https://www.kaggle.com/datasets/gpiosenka/moths-image-datasetclassification)
- Method:
    - Option 1: Download manually and unzip into `data/kaggle-dataset/`
    - Option 2: Use `kagglehub` or the Kaggle API:
      ```python
      import kagglehub
      path = kagglehub.dataset_download("gpiosenka/moths-image-datasetclassification")
      ```

Ensure all paths in your notebooks use relative paths like:
```python
from pathlib import Path
DATA_DIR = Path("../data/kaggle-dataset/")
