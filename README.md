# Road Segmentation in Satellite Images

## Overview
In the evolving landscape of digital image processing and computer vision, our project focuses on road segmentation from satellite images. We leverage state-of-the-art architectures like U-Net and DeepLabV3 for effective image segmentation.

## Models
### DeepLabV3 (ResNet101)
- **File**: [models/deeplabv3.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/DeepLabV3.py) - Wrapper class for deeplabv3_resnet50 by PyTorch. 
- **Notebook**: [notebook/deeplab-v3.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/deeplab-v3.ipynb)
- **Description**: Implements the DeepLabV3 model with a ResNet50 backbone.

### UNet Variants
- **Files**: [models/UNetV1.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/UNetV1.py), [models/UNetV2.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/UNetV2.py), [models/UNetV3.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/UNetV3.py) - Wrapper classes for each UNet.
- **Notebooks** [notebooks/unet-v1.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/unet-v1.ipynb), [notebooks/unet-v2.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/unet-v2.ipynb), [notebooks/unet-v3.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/unet-v3.ipynb)
- **Description**: Different versions of the UNet model, each tailored for road segmentation. Implementation shown in notebooks.

## Datasets
1. **AIcrowd Dataset**: High-resolution satellite images with labeled roads.
2. **Massachusetts Roads Dataset**: 1500x1500 pixel images, segmented into smaller parts. See [notebook](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/massachusetts.ipynb) for preprocessing this dataset for our needs. 
4. **Kaggle Dataset**: 400x400 pixel images from Los Angeles, filtered for road presence. See [notebook](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/kaggle.ipynb) for preprocessing this dataset for our needs.

## Installation
- **Requirements**: Python 3.x, PyTorch, torchvision, timm, matplotlib, omegaconf, wandb
- **Setup**: Run `pip install -r requirements.txt` to install dependencies. [View requirements](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/requirements.txt)

## Configuration
- **File**: [config.yaml](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/config.py)
- **Description**: Contains settings for model parameters, training settings, and data paths.

## Usage
- **Training**: To train reproduce the pipeline for training the best model execute:
  ```bash
  TODO
- **Testing**: To reproduce the best result execute:
  ```bash
  python run.py

## Evaluation
- **Metric**: F1 Score
- **Description**: The F1 Score is used to evaluate model performance, balancing precision and recall.

---

Note: For specific implementation details, refer to the comments and documentation within each code file.
