from keras.applications import VGG16

from src.cbr_model import CaseBasedReasoning
from src.cbr_model.feature_extraction import nn_image_to_vector
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
