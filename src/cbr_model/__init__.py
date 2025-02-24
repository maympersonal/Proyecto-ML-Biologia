import os
import cv2
import pickle

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from src.cbr_model.knowledge_managers import knn_manager
from src.utils.file_pipeline import LabeledImageRef
from src.utils.image_pipeline import (
    detect_edges,
    find_bounding_boxes,
)


class CaseBasedReasoning:
    def __init__(self, k: int = 20) -> None:
        self.k = k
        self.feature_extractor: Any = lambda image: None
        self.distance_measurement: Any = lambda a, b: np.linalg.norm(a - b)
        self.knowledge_manager: Any = knn_manager
        self.segment_image: Any = lambda image: find_bounding_boxes(detect_edges(image))

    def fit(
        self,
        input_data: Iterable[LabeledImageRef],
        labels: Dict[int, Any] | None = None,
        default_label: str = "default",
    ) -> None:
        if not labels:
            labels = dict()

        self.classifications = []
        n = len(list(input_data))

        for image_path, label_path in input_data:
            print(f"Image {image_path} processed. {n} left.")
            n -= 1

            image = cv2.imread(image_path)  # type: ignore

            height, width, _ = image.shape

            image_labels = self._extract_labels(label_path, height, width)

            for class_id, label in image_labels:
                crop = self._crop_image(image, label)

                features = self.feature_extractor(crop)

                self.classifications.append(
                    (features, labels.get(class_id, default_label))
                )

    def predict(self, images: List[str]) -> List[str]:
        predictions = []

        for image_dir in images:
            print(image_dir)
            image = cv2.imread(image_dir)  # type: ignore
            boxes = self.segment_image(image)

            if not boxes:
                continue

            image_predictions = []

            crops = [self._crop_image(image, box) for box in boxes]

            for crop, box in zip(crops, boxes):
                crop_features = self.feature_extractor(crop)

                distances = [
                    (self.distance_measurement(crop_features, features), clasif)
                    for features, clasif in self.classifications
                ]

                predicted = self.knowledge_manager(distances, self.k)

                if predicted:
                    image_predictions.append((box, predicted))

            predictions.append(image_predictions)
            print(image_predictions)

        return predictions

    def save_to(self, output_dir: str) -> None:
        if not os.path.exists(os.path.dirname(output_dir)):
            os.makedirs(os.path.dirname(output_dir))

        with open(output_dir, "wb") as f:
            pickle.dump(self.classifications, f)

    def load_from(self, output_file: str) -> None:
        if not os.path.isfile(output_file):
            raise FileNotFoundError(f"File {output_file} not found.")

        with open(output_file, "rb") as f:
            self.classifications = pickle.load(f)

    def _extract_labels(self, label_path, height, width):
        with open(label_path, "r") as label_file:
            image_labels_raw = map(
                lambda t: map(float, t),
                [line.strip().split() for line in label_file.readlines()],
            )

            image_labels = [
                (
                    int(id),
                    (
                        max(0, int((x - w / 2) * width) - 10),
                        max(0, int((y - h / 2) * height) - 10),
                        int(width * w) + 20,
                        int(height * h) + 20,
                    ),
                )
                for id, x, y, w, h in image_labels_raw
            ]
        return image_labels

    def _crop_image(self, image, label: Tuple[float, float, float, float]):
        x, y, width, height = label
        return image[y : y + height, x : x + width]
