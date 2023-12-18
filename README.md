# Road Segmentation in Satellite Images

## Overview
In the evolving landscape of digital image processing and computer vision, our project focuses on road segmentation from satellite images. We leverage state-of-the-art architectures like U-Net and DeepLabV3 for effective image segmentation.

## Models
### DeepLabV3 (ResNet101)
- **File**: [models/deeplabv3.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/deeplabv3.py)
- **Description**: Implements the DeepLabV3 model with a ResNet50 backbone.
![deeplab_v3](https://github.com/ilievanadezhda/Road-Segmentation-ML/assets/58995762/3c621301-3465-43ef-ba65-192e055699c8)

### UNet Variants
- **Files**: [models/unet_v1.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/unet_v1.py), [models/unet_v2.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/unet_v2.py), [models/unet_v3.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/unet_v3.py)
- **Description**: Different versions of the UNet model, each tailored for road segmentation.
![unet](https://github.com/ilievanadezhda/Road-Segmentation-ML/assets/58995762/892d35e0-216b-4122-9613-d78076422751)

## Datasets
1. **AIcrowd Dataset**: High-resolution satellite images with labeled roads.
2. **Massachusetts Roads Dataset**: 1500x1500 pixel images, segmented into smaller parts.
3. **Kaggle Dataset**: 400x400 pixel images from Los Angeles, filtered for road presence.

## Installation
- **Requirements**: Python 3.x, PyTorch, torchvision, timm, matplotlib, omegaconf, wandb
- **Setup**: Run `pip install -r requirements.txt` to install dependencies. [View requirements](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/requirements.txt)

## Configuration
- **File**: [config.yaml](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/config.py)
- **Description**: Contains settings for model parameters, training settings, and data paths.

## Usage
- **Training**: To train reproduce the pipeline for training the best model execute:
  ```bash
  pyhon train.py
- **Testing**: To reproduce the best result execute:
  ```bash
  python run.py

## Evaluation
- **Metric**: F1 Score
- **Description**: The F1 Score is used to evaluate model performance, balancing precision and recall.

---

Note: For specific implementation details, refer to the comments and documentation within each code file.
