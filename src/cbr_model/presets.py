from keras.applications import VGG16

from src.cbr_model import CaseBasedReasoning
from src.cbr_model.feature_extraction import nn_image_to_vector
from src.cbr_model.feature_extraction_autoencoder import autoencoder_image_to_vector
from src.cbr_model.feature_extraction_SIFT_LBP import sift_lbp_image_to_vector
from src.cbr_model.knowledge_managers_svm import svm_manager
from src.cbr_model.knowledge_managers_random_forest import random_forest_manager
from src.cbr_model.build_autoencoder import get_autoencoder
from src.utils.image_pipeline import resize_and_pad_image

"""
The names of the presets are as follows:

1. Feature extraction method
2. Distance measurement method (default is euclidean distance)
3. Knowledge manager
4. Image segmentation method (if omitted, the canny filter is used)
"""


def imagenetCNN_euclideanDistance_knn() -> CaseBasedReasoning:
    cbr = CaseBasedReasoning()

    model = VGG16(weights="imagenet", include_top=False, pooling="avg")

    cbr.feature_extractor = lambda image: nn_image_to_vector(
        resize_and_pad_image(image), model
    )

    return cbr


def imagenetCNN_euclideanDistance_autoencoder_svm(encoder) -> CaseBasedReasoning:
    cbr = CaseBasedReasoning()

    cbr.feature_extractor = lambda image: autoencoder_image_to_vector(
        resize_and_pad_image(image), encoder
    )

    cbr.knowledge_manager = svm_manager

    return cbr


def imagenetCNN_euclideanDistance_sift_lbp_random_forest() -> CaseBasedReasoning:
    cbr = CaseBasedReasoning()

    model = VGG16(weights="imagenet", include_top=False, pooling="avg")

    cbr.feature_extractor = lambda image: sift_lbp_image_to_vector(
        resize_and_pad_image(image), model
    )

    cbr.knowledge_manager = random_forest_manager

    return cbr
