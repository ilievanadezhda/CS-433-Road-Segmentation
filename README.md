# Road-Segmentation-ML

## Overview
"Road-Segmentation-ML" is a machine learning project aimed at segmenting roads in satellite images using deep learning techniques. The project implements various models to accurately classify each pixel in an image as road or non-road.

## Models
This project includes several deep learning models, each with its unique approach to the segmentation task:

### DeepLabV3 (ResNet101)
- **File**: `models/deeplabv3.py`
- **Description**: Implements the DeepLabV3 model with a ResNet101 backbone.
- **Main Function**: Provides a robust framework for pixel-level image segmentation.

### UNet Variants
- **Files**: `models/unet_v1.py`, `models/unet_v2.py`, `models/unet_v3.py`
- **Description**: These files contain different versions of the UNet model, each tailored for road segmentation.
- **Main Function**: UNet models are designed for biomedical image segmentation but are adapted here for high-resolution satellite image segmentation, offering precise pixel classification.

## Dataset
- **Description**: The dataset comprises high-resolution satellite images with corresponding labeled images marking roads.
- **Source**: [AICrowd Road Segmentation Challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation)
- **Usage**: Used for training and evaluating the models, providing a benchmark for segmentation accuracy.

## Installation
- **Requirements**: Python 3.x, PyTorch, torchvision, timm, matplotlib, omegaconf, wandb
- **Setup**: Run `pip install -r requirements.txt` to install dependencies.

## Configuration
- **File**: `config.yaml`
- **Description**: Contains configuration settings for the project, including model parameters, training settings, and data paths.
- **Usage**: Customize the training and model parameters by editing this file.

## Jupyter Notebooks
These notebooks provide practical examples and visualizations:

### DeepLabV3 Notebook
- **File**: `deeplab-v3.ipynb`
- **Purpose**: Demonstrates the application of the DeepLabV3 model on the dataset.

### UNet Notebooks
- **Files**: `unet-v1.ipynb`, `unet-v2.ipynb`, `unet-v3.ipynb`
- **Purpose**: Showcases the usage and performance of different UNet variants.

### Visualization Notebook
- **File**: `visualization.ipynb`
- **Purpose**: Used for visualizing the segmentation results, comparing ground truth with predictions.

## Evaluation
- **Metric**: F1 Score
- **Description**: The F1 Score is used as the primary metric to evaluate the performance of the models, balancing precision and recall.

---

Note: This README provides an overview of the project. For specific implementation details, refer to the comments and documentation within each code file.
