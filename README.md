# Road Segmentation in Satellite Images

## Overview
In the evolving landscape of digital image processing and computer vision, road segmentation from satellite images stands as a challenging domain. Road segmentation is a process used in computer vision and image processing where the goal is to identify and isolate the parts of an image that represent roads. This project dives deep into the process of segmenting roads from satellite images. We leverage state-of-the-art architectures such as U-Net and DeepLabV3 for effective image segmentation.

## Models
#### DeepLabV3
- **File**: Wrapper class for deeplabv3_resnet50 by PyTorch. Located: [models/deeplabv3.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/DeepLabV3.py) 
- **Notebook**: [notebook/deeplab-v3.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/deeplab-v3.ipynb)
- **Description**: Implements the DeepLabV3 model with a ResNet50 backbone.

#### UNet(s)
- **Files**: Wrapper classes for each UNet. Located: [models/UNetV1.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/UNetV1.py), [models/UNetV2.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/UNetV2.py), [models/UNetV3.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/models/UNetV3.py) 
- **Notebooks**: [notebooks/unet-v1.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/unet-v1.ipynb), [notebooks/unet-v2.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/unet-v2.ipynb), [notebooks/unet-v3.ipynb](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/unet-v3.ipynb)
- **Description**: Different versions of the UNet model, each tailored for road segmentation. Implementation shown in notebooks.

## Datasets
1. **AIcrowd Dataset**: High-resolution satellite images with labeled roads.
2. **Massachusetts Roads Dataset**: 1500x1500 pixel images, segmented into smaller parts. See [notebook](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/massachusetts.ipynb) for preprocessing this dataset for our needs. [Link](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset) to original dataset.
4. **Kaggle Dataset**: 400x400 pixel images from Los Angeles, filtered for road presence. The original dataset was downloaded using Googlemaps API. See [notebook](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/notebooks/kaggle.ipynb) for preprocessing this dataset for our needs. [Link](https://www.kaggle.com/datasets/timothlaborie/roadsegmentation-boston-losangeles) to original dataset.

## Installation
Note: This guide is for users who have anaconda or miniconda installed. If you are using a different tool for managing environments such as `venv` then skip steps [2-4] and create the environment following the [guideline](https://docs.python.org/3/library/venv.html)
1. Clone the repository:
   ```bash
   git clone ...
   cd ...
2. Create an environment using Python 3.8.18
   ```bash
   conda create --name road_segmentation python==3.8.18
3. Activate the environment
   ```bash
   conda activate road_segmentaiton
4. Install the required packages:
   ``` bash
   pip install -r requirements.txt

## Usage
- **Training**: See [training_pipeline](), this script takes 12 hours to run on a NVIDIA GeForce RTX 3050 Ti (laptop version). To reproduce the best model checkpoint execute:
  ```bash
  pyton training_pipeline.py
- **Testing**: For ease of use, we provide the best models' checkpoint. To reproduce the best result execute:
  ```bash
  python run.py

## Additional
- **Configuration File**: Contains settings for model parameters, training settings, and data paths. [config.yaml](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/config.py)
- **Postprocessing**: Contains postprocessing functions. [postprocessing.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/postprocessing.py)
- **Utils**: Utility functions for training and evaluation. [train_utils.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/train_utils.py), [utils.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/utils.py)
- **Baseline**: Modified `tf_aerial_images.py`, which demonstrates the use of a basic convolutional neural network in TensorFlow for generating a baseline. See [tf_aerial_images.py](https://github.com/ilievanadezhda/Road-Segmentation-ML/blob/main/examples/tf_aerial_images.py). In order to run this script you need to install tensorflow==2.11.0. To avoid environment conflicts we recommend you to create a new environment and install this dependency separately. 

## Ethical Considerations

In the development and application of our road segmentation project, we recognize the importance of addressing ethical concerns. This section highlights key areas of focus and our recommendations for responsible use.

#### Bias and Inaccuracy

- **Description**: Machine learning models, including those used in road segmentation, can inherit biases present in their training data. This can lead to inaccuracies, particularly in underrepresented areas, affecting the model's performance and fairness.
- **Mitigation Strategies**: 
  - **Diverse Data Sets**: We recommend using diverse and comprehensive datasets that represent various geographical and environmental conditions to minimize bias.
  - **Continuous Evaluation**: Regularly evaluate and update the model to ensure it remains accurate and fair across different regions and conditions.

#### Surveillance and Misuse

- **Description**: There is a potential risk that road segmentation technology could be used for surveillance purposes or other forms of misuse, especially if integrated with real-time monitoring systems.
- **Mitigation Strategies**:
  - **Ethical Usage Guidelines**: We advocate for the establishment of clear ethical guidelines governing the use of road segmentation technology, particularly in sensitive applications.

#### Conclusion

We are committed to the responsible development and use of AI technologies. We encourage users of our road segmentation project to consider these ethical aspects and to apply the technology in a manner that respects privacy and promotes fairness and safety.



---
