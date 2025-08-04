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
