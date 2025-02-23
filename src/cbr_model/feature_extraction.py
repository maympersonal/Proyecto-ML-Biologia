import cv2
import numpy as np

from src.utils.image_pipeline import (
    difference_of_gaussians,
    to_grayscale,
    add_images,
)

from keras.applications.vgg16 import preprocess_input
from keras.models import Model


def hull_processing(image):
    """This method is explained on the paper 9 but only works on tinted images :("""
    gray = to_grayscale(image)
    diff = difference_of_gaussians(gray)
    edges = cv2.Canny(add_images(gray, diff), 50, 150)  # type: ignore

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # type: ignore
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)  # type: ignore
    hull = cv2.convexHull(largest_contour)  # type: ignore

    rect = cv2.minAreaRect(hull)  # type: ignore
    box = cv2.boxPoints(rect)  # type: ignore
    box = np.int0(box)

    angle = rect[2]
    if angle < -45:
        angle += 90

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # type: ignore
    rotated = cv2.warpAffine(image, M, (w, h))  # type: ignore

    return rotated, hull


def draw_polygon_on_image(image, hull):
    cv2.drawContours(image, [hull], -1, (255, 0, 0), 1)  # type: ignore
    return image


def nn_image_to_vector(image: np.ndarray, model: Model) -> np.ndarray:
    if image.shape[:2] != (224, 224):
        raise ValueError("Image must be of size 224x224 for VGG16 model.")

    image_array = np.expand_dims(image, axis=0)
    preprocessed_image = preprocess_input(image_array)
    features = model.predict(preprocessed_image)
    return features.flatten()
