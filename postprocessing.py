from skimage.morphology import (
    square,
    opening,
    erosion,
)


def apply_morphological_operations(prediction):
    # opening to the combined image to remove small white spots
    cleaned_prediction = opening(prediction, square(3))

    # erosion to thin up the roads
    thinned_prediction = erosion(cleaned_prediction, square(4))

    return thinned_prediction
