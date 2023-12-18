import os
import torch
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
from datasets.TestDataset import TestDataset
from models.DeepLabV3 import ResNet50
from examples.mask_to_submission import *
from postprocessing import apply_morphological_operations
from skimage.morphology import square, opening, erosion


def apply_morphological_operations(prediction):
    """Applies morphological operations to the prediction.

    Args:
        prediction: predicted labels

    Returns:
        thinned_prediction: prediction after morphological operations
    """
    # opening to the combined image to remove small white spots
    cleaned_prediction = opening(prediction, square(3))
    # erosion to thin up the roads
    thinned_prediction = erosion(cleaned_prediction, square(9))

    return thinned_prediction


def load_checkpoint(model_path):
    try:
        checkpoint = torch.load(model_path)
        print("Checkpoint loaded successfully.")
    except:
        print("Loading checkpoint failed. Trying to load it with map_location.")
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    return checkpoint


def create_model(checkpoint):
    model = ResNet50()
    model.load_state_dict(checkpoint)
    print("Model created and weights loaded.")
    return model


def get_transform():
    mean = [0.3353, 0.3328, 0.2984]
    std = [0.1967, 0.1896, 0.1897]
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def save_prediction(prediction, index, output_folder):
    prediction_filename = f"{output_folder}/prediction_{index + 1}.png"
    plt.imsave(prediction_filename, prediction.squeeze(), cmap="gray")
    print(f"Saved prediction {index + 1}")
    return prediction_filename


def predict_and_save(model, test_dataset, output_folder, threshold=0.6):
    prediction_filenames = []
    for i, image in enumerate(test_dataset):
        with torch.no_grad():
            prediction = model(image.unsqueeze(0))
        prediction = torch.sigmoid(prediction)
        prediction = prediction.squeeze().detach().numpy()
        prediction = apply_morphological_operations(prediction)
        prediction = (prediction > threshold).astype(int)
        prediction_filename = "postprocess/prediction_" + str(i + 1) + ".png"
        prediction_filenames.append(prediction_filename)
    return prediction_filenames


def main():
    MODEL = "models/checkpoints/deeplabv3_resnet50_HUGE_retrain.pt"
    TEST_FOLDER = "datasets/test/"
    OUTPUT_FOLDER = "predictions"

    print("Starting the segmentation process...")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    checkpoint = load_checkpoint(MODEL)
    model = create_model(checkpoint)
    transform = get_transform()
    test_dataset = TestDataset(TEST_FOLDER, transform=transform)
    print(f"Loaded {len(test_dataset)} images from test dataset.")

    model.eval()
    prediction_filenames = predict_and_save(model, test_dataset, OUTPUT_FOLDER)

    masks_to_submission("submission.csv", *prediction_filenames)
    print("All predictions saved. Submission file created.")


if __name__ == "__main__":
    main()
