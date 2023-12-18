# Road Segmentation in Satellite Images

## Overview
In the evolving landscape of digital image processing and computer vision, our project focuses on road segmentation from satellite images. We leverage state-of-the-art architectures like U-Net and DeepLabV3 for effective image segmentation.

## Models
#### DeepLabV3
- **File**: Wrapper class for deeplabv3_resnet50 by PyTorch. Located: [models/deeplabv3.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/DeepLabV3.py) 
- **Notebook**: [notebook/deeplab-v3.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/deeplab-v3.ipynb)
- **Description**: Implements the DeepLabV3 model with a ResNet50 backbone.

#### UNet(s)
- **Files**: Wrapper classes for each UNet. Located: [models/UNetV1.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/UNetV1.py), [models/UNetV2.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/UNetV2.py), [models/UNetV3.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/UNetV3.py) 
- **Notebooks** [notebooks/unet-v1.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/unet-v1.ipynb), [notebooks/unet-v2.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/unet-v2.ipynb), [notebooks/unet-v3.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/unet-v3.ipynb)
- **Description**: Different versions of the UNet model, each tailored for road segmentation. Implementation shown in notebooks.

## Datasets
1. **AIcrowd Dataset**: High-resolution satellite images with labeled roads.
2. **Massachusetts Roads Dataset**: 1500x1500 pixel images, segmented into smaller parts. See [notebook](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/massachusetts.ipynb) for preprocessing this dataset for our needs. 
4. **Kaggle Dataset**: 400x400 pixel images from Los Angeles, filtered for road presence. See [notebook](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/kaggle.ipynb) for preprocessing this dataset for our needs.

## Installation
- **Requirements**: Python 3.x, PyTorch, torchvision, timm, matplotlib, omegaconf, wandb
- **Setup**: Run `pip install -r requirements.txt` to install dependencies. [View requirements](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/requirements.txt)

## Additional
- **Configuration File**: Contains settings for model parameters, training settings, and data paths. [config.yaml](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/config.py)
- **Postprocessing**: Contains postprocessing functions. [postprocessing.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/postprocessing.py)
- **Utils**: Utility functions for training and evaluation. [train_utils.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/train_utils.py), [utils.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/utils.py)
- **Baseline**: Modified `tf_aerial_images.py`, which demonstrates the use of a basic convolutional neural network in TensorFlow for generating a baseline. See [tf_aerial_images.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/examples/tf_aerial_images.py). In order to run this script you need to install tensorflow==2.11.0. In order to avoid environment conflicts we recommend you to create a new environment and install this dependency separately. 

## Usage
- **Training**: See [training_pipeline](), this script takes 12 hours to run on a NVIDIA GeForce RTX 3050 Ti (laptop version). To reproduce the best model checkpoint execute:
  ```bash
  pyton training_pipeline.py
- **Testing**: To reproduce the best result execute:
  ```bash
  python run.py


---

Note: For specific implementation details, refer to the comments and documentation within each code file.
