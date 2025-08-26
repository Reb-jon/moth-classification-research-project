## Deep Learning Approaches for Moth Species Classification and Conservation Monitoring
[![DOI](https://zenodo.org/badge/1001463270.svg)](https://doi.org/10.5281/zenodo.16949893)

This repository contains the code, results, and visualisations for the MSc Data Science Extended Research Project at the University of Manchester investigating automated moth species classification using deep learning architectures.

### Project Overview
This study develops a systematic methodology using deep learning for automated moth species classification to identify optimal architectures and assess deployment suitability for conservation monitoring practices. Seven architectures including CNNs, Vision Transformers, and hybrid models were evaluated on 50 moth species using a four-phase optimisation pipeline.

### Repository Structure
```
moth-classification-research-project/
├── notebooks/                    
│   ├── 01_resnet50_baseline.ipynb                       # Frozen ResNet50 control
│   ├── 02_resnet50_partial_ft.ipynb                     # ResNet50 with partial fine-tuning
│   ├── 03_efficientnet_b0_partial_ft.ipynb              # EfficientNet_B0 baseline
│   ├── 04_inception_v3_partial_ft.ipynb                 # InceptionV3 baseline
│   ├── 05_vgg16_partial_ft.ipynb                        # VGG16 baseline
│   ├── 06_mobilenet_v2_partial_ft.ipynb                 # MobileNetV2 baseline
│   ├── 07_vit_b16_partial_ft.ipynb                      # ViT-B/16 baseline
│   ├── 08_coat_lite_medium_partial_ft.ipynb             # CoaT-Lite Medium baseline
│   ├── 11_efficientnet_full_ft.ipynb                    # EfficientNet_B0 optimisation pipeline
│   ├── 12_mobilenet_v2_full_ft.ipynb                    # MobileNetV2 optimisation pipeline
│   ├── 13_vit_b16_full_ft.ipynb                         # ViT-B/16 optimisation pipeline
│   ├── 14_coat_lite_medium_full_ft.ipynb                # CoaT-Lite optimisation pipeline
│   ├── confused-species-extra-transformations.ipynb     # Targeted augmentation application
│   ├── data-exploration-and-figures.ipynb               # Dataset analysis & thesis figure creation
│   └── *.pth files                                      # Trained model weight details
├── figures/                           # Generated figures and plots
├── tables/                            # Results tables and summaries
├── cam_outputs/                       # Score-CAM & Grad-CAM++ visualisations
│   ├── efficientnet_b0/                                 # EfficientNet_B0 Baseline: Grad-CAM++ (baseline)
│   ├── efficientnet_b0_ft/                              # EfficientNet_B0 Fine-tuned: Grad-CAM++ (optimisation stages)
│   ├── mobilenetv2/                                     # MobileNetV2 Baseline: Score-CAM (baseline) 
│   ├── mobilenetv2_ft/                                  # MobileNetV2 Fine-tuned: Score-CAM (optimisation stages)
│   ├── vit_b16/                                         # ViT-B/16 Baseline: Score-CAM (baseline)
│   ├── vit_b16_ft/                                      # ViT-B/16 Fine-tuned: Score-CAM (optimisation stages)
│   ├── coat_lite/                                       # CoaT-Lite Medium Baseline: Score-CAM (baseline)
│   └── coat_lite_ft/                                    # CoaT-Lite Medium Fine-tuned: Score-CAM (optimisation stages)
├── data/                              # Dataset directory
│   └── kaggle-dataset/                                  # Moth images dataset
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── .gitignore                         # Git ignore rules
└── README.md                          # README file
```
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
```bash
git clone https://github.com/Reb-jon/moth-classification-research-project.git
cd moth-classification-research-project
```
#### *** Recommended: Create dedicated conda environment 
```bash
conda env create -f environment.yml
conda activate moth-classification
```
#### Alternative: pip installation
```python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## Dataset Setup

This project uses one dataset. Please download the dataset manually and place it in the `data/` folder.

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
```

### Dataset Structure
After extraction, your data folder should look like:
```
data/
└── kaggle-dataset/
    ├── train/
    │   ├── species_1/
    │   ├── species_2/
    │   └── ...
    ├── valid/
    │   ├── species_1/
    │   ├── species_2/
    │   └── ...
    └── test/
    │   ├── species_1/
    │   ├── species_2/
    │   └── ...
```
## Usage
### Quick Start
#### Phase 1: Baseline Model Evaluation
1. Run notebooks 01-08 to reproduce the seven architecture comparison
2. Review baseline performance and select top four models

#### Phase 2: Optimisation Pipeline
3. Run notebooks 11-14 for the systematic four-phase optimization (CutMix → Standard → Targeted → Fine-tuning)
4. confused-species-extra-transformations.ipynb documents the targeted augmentation strategies

#### Phase 3: Analysis and Visualisation
5. data-exploration-and-figures.ipynb contains dataset analysis overview and thesis figure creation
6. Trained model weights (*.pth files) are saved in the notebooks directory

##  Technical Requirements
### Dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
pillow>=8.0.0
pytorch-grad-cam>=1.3.0

### Hardware Requirements
- Python: 3.12.3
- Minimum: 8GB RAM, CPU (training will be slow but functional)
- *** Note: This project was developed and tested using CPU-only PyTorch. Training times in the thesis reflect CPU performance.

## Project Outputs
### Generated Files
- figures/: All plots, charts, and visualisations from the analysis
- tables/: Performance metrics, confusion matrices, and result summaries
- cam_outputs/: Score-CAM and Grad-CAM++ attention visualisations
- notebooks/: Complete analysis pipeline with results
  
## Citation
If you use this repository, please cite it as:

Rebecca Jones. (2025). *Deep Learning Approaches for Moth Species Classification and Conservation Monitoring* (v1.0.1). University of Manchester. Zenodo. https://doi.org/10.5281/zenodo.16949894

## License
This project is for academic purposes. Dataset usage is subject to original licensing terms.

## Acknowledgments
- Supervisor: Professor David Topping
- Dataset: Gerry's Moths Image Dataset (Kaggle)
- University of Manchester School of Social Sciences
