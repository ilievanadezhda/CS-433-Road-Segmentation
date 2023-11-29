import cv2
import numpy as np


def postprocess(image):
    # Noise reduction (using morphological opening)
    kernel_op = np.zeros((5, 5), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_op, iterations=3)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_op, iterations=3)

    # kernel_dil = np.zeros((3, 3), np.uint8)
    # Background area determination (Dilation to increase the background area)
    # sure_bg = cv2.dilate(opening, kernel_dil, iterations=3)

    # # Identifying sure foreground area (roads)
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L1, 5)
    # _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)= 0

    return closing


def apply_morphological_ops(segmentation_map):
    # define kernel
    kernel = np.ones((3, 3), np.uint8)

    # erosion
    erosion = cv2.erode(segmentation_map, kernel, iterations=4)

    # dilation
    dilation = cv2.dilate(erosion, kernel, iterations=4)

    return dilation


# processed_map = apply_morphological_ops(segmentation_map)
