"""CNN Model for Baseline Solution for Aerial Imagery Segmentation."""

import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

# Constants
NUM_CHANNELS = 3
NUM_LABELS = 2
TRAINING_SIZE = 100
VALIDATION_SIZE = 0
IMG_PATCH_SIZE = 16
BATCH_SIZE = 16
NUM_EPOCHS = 100

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!


def img_crop(im, w, h):
    """Crop an image into patches of size w x h"""
    list_patches = []
    imgwidth, imgheight = im.shape[0], im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_data(filename, num_images):
    """Extract images."""
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print(f"File {image_filename} does not exist")

    img_patches = [
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(len(imgs))
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]
    return np.asarray(data)


def value_to_class(v):
    """Assign a label to a patch v."""
    foreground_threshold = 0.25
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def extract_labels(filename, num_images):
    """Extract labels."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print(f"File {image_filename} does not exist")

    gt_patches = [
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE)
        for i in range(len(gt_imgs))
    ]
    data = [
        gt_patches[i][j]
        for i in range(len(gt_patches))
        for j in range(len(gt_patches[i]))
    ]
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
    return labels


def create_model():
    """Create CNN model"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32,
                (5, 5),
                activation="relu",
                input_shape=(IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(NUM_LABELS, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def load_test_patches(test_data_dir, num_test_images):
    """Load test patches"""
    test_patches = []
    for i in range(1, num_test_images + 1):
        folder_name = f"test_{i}"
        image_filename = os.path.join(test_data_dir, folder_name, folder_name + ".png")
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            patches = img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
            test_patches.extend(patches)  # append patches
        else:
            print(f"File {image_filename} does not exist")
    return np.asarray(test_patches)


def save_predictions_to_csv(predictions, num_test_images, original_size, csv_filename):
    """Make a submission CSV file."""
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "prediction"])

        patch_size = IMG_PATCH_SIZE
        num_patches_per_side = original_size // patch_size
        patch_index = 0

        for image_index in range(1, num_test_images + 1):
            for row in range(0, original_size, patch_size):
                for col in range(0, original_size, patch_size):
                    patch_id = f"{str(image_index).zfill(3)}_{row}_{col}"
                    # get the prediction for this patch
                    patch_prediction = np.argmax(predictions[patch_index])
                    writer.writerow([patch_id, patch_prediction])
                    patch_index += 1


def main():
    data_dir = "../datasets/train/"
    train_data_filename = data_dir + "images/"
    train_labels_filename = data_dir + "groundtruth/"

    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    model_dir = "baseline_model/"
    # check if already trained
    if not os.path.isdir(model_dir):
        model = create_model()
        model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
        # create dir
        os.mkdir(model_dir)
        model.save(model_dir + "baseline_model.h5")
    else:
        model = tf.keras.models.load_model(model_dir + "baseline_model.h5")

    test_data_dir = "../datasets/test/"  # Update this path as necessary
    num_test_images = 50  # Update based on the number of test images you have
    test_data = load_test_patches(test_data_dir, num_test_images)

    # make predictions on test data patches
    test_patch_predictions = model.predict(test_data)

    csv_filename = "submission.csv"
    save_predictions_to_csv(test_patch_predictions, num_test_images, 608, csv_filename)


if __name__ == "__main__":
    main()
