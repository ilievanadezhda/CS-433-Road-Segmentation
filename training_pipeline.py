"""
This script is used to train the best model. It first trains the model with the initial 
training configuration containing the 3 datasets, and then retrains it with the retraining 
configuration containing only the AIcrowd dataset. The best model is saved as models/checkpoints/deeplabv3_resnet50_best.pt.
"""

import os
import torch
import gc
import zipfile
import requests
from omegaconf import OmegaConf
from utils import set_seeds
from train_utils import prepare_data, prepare_model, prepare_optimizer, train
from utils import set_seeds
import warnings

warnings.filterwarnings("ignore")

set_seeds()

# define constants
MODEL_DIR = "models/checkpoints"
INITIAL_TRAIN_CONFIG = "config/initial_train_config.yaml"
RETRAIN_CONFIG = "config/retrain_config.yaml"
MODEL_CHECKPOINT = "models/checkpoints/deeplabv3_resnet50_large.pt"
BEST_MODEL_CHECKPOINT = "models/checkpoints/deeplabv3_resnet50_best.pt"


def download_data():
    """Download massachusetts and kaggle datasets."""
    # if folders datasets/kaggle and datasets/massachusetts_384 do not exist, download the datasets
    if not os.path.exists("datasets/kaggle/") or not os.path.exists(
        "datasets/massachusetts_384/"
    ):
        # dropbox URL
        dropbox_url = "https://www.dropbox.com/scl/fi/vl5ygbyuyhchcp7getra7/datasets.zip?rlkey=avjkd4q1x0v7dilid56h0fdy8&dl=1"

        # file path where the .zip file will be saved
        file_path = "datasets/datasets.zip"

        response = requests.get(dropbox_url)

        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            message = "Download successful. The file has been saved as 'datasets.zip'."
        else:
            message = "Failed to download the file. Error code: " + str(
                response.status_code
            )

        print(message)

        # path to the downloaded .zip file
        zip_file_path = "datasets/datasets.zip"

        # directory to extract the contents of the zip file, in this current folder
        extraction_path = "datasets/"

        # unzipping the file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extraction_path)

        extraction_message = (
            f"The contents of the zip file have been extracted to: {extraction_path}"
        )

        print(extraction_message)

        # delete the datasets.zip file
        os.remove(zip_file_path)

    else:
        print("Datasets already downloaded.")


def create_directory(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_configuration(file_path):
    """Load configuration from a YAML or JSON file."""
    print(f"Loading configuration from {file_path}")
    return OmegaConf.load(file_path)


def initialize_training(config):
    """Initialize and return the necessary components for training."""
    print("Initializing training...")
    # set seeds for reproducibility
    set_seeds(config.seed)

    # prepare data loaders
    train_loader, val_loader = prepare_data(config)

    # prepare model
    model = prepare_model(config)

    # define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = prepare_optimizer(model, config)

    # set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Training initialized successfully.")
    return model, train_loader, val_loader, criterion, optimizer, device


def main():
    # download the datasets
    print("Downloading datasets...")
    download_data()

    print("Starting the training process...")

    # create model directory
    create_directory(MODEL_DIR)

    # load initial training configuration and initialize training components
    print("Starting initial training phase...")
    initial_config = load_configuration(INITIAL_TRAIN_CONFIG)
    model, train_loader, val_loader, criterion, optimizer, device = initialize_training(
        initial_config
    )

    # check if wandb should be used for initial training, default is False
    if not initial_config.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    # train the model
    print("Training model...")
    trained_model = train(
        model, device, train_loader, val_loader, criterion, optimizer, initial_config
    )
    print("Initial training phase completed.")

    # clear cache
    print("Clearing cache...")
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared %d bytes from cache." % gc.collect())

    # load retraining configuration
    print("Starting retraining phase...")
    retrain_config = load_configuration(RETRAIN_CONFIG)

    # load the best model for retraining
    print(f"Loading model from {MODEL_CHECKPOINT} for retraining...")
    model.load_state_dict(torch.load(MODEL_CHECKPOINT))
    model.to(device)

    # update training components for retraining
    _, train_loader, val_loader, criterion, optimizer, _ = initialize_training(
        retrain_config
    )

    # check if wandb should be used for retraining, default is False
    if not retrain_config.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.pop("WANDB_DISABLED", None)

    # retrain the model
    print("Retraining model...")
    retrained_model = train(
        model, device, train_loader, val_loader, criterion, optimizer, retrain_config
    )
    print("Retraining phase completed.")

    print(f"Model trained successfully. Saved as {BEST_MODEL_CHECKPOINT}")


if __name__ == "__main__":
    main()
