import os
from typing import List, Optional, Tuple
import yaml
from pydantic import BaseModel
import random


class YOLODataModel(BaseModel):
    path: Optional[str] = None
    train: Optional[str] = None
    val: Optional[str] = None
    test: Optional[str] = None
    nc: int
    names: List[str]


class LabeledImageRef(BaseModel):
    image_path: str
    label_path: str


class FilePipeline:
    def __init__(self, yolo_yaml: str):
        with open(yolo_yaml, "r") as file:
            data = yaml.safe_load(file)
            obj_data = YOLODataModel(**data)
            self.params = obj_data

        self.base_dir = os.path.dirname(yolo_yaml)
        self.base_dir = os.path.join(
            self.base_dir, obj_data.path if obj_data.path else "./"
        )

        self.labeled_images = self._collect_labeled_images()

    def _collect_labeled_images(self) -> List[LabeledImageRef]:
        labeled_images = []
        for split in ["train", "val", "test"]:
            split_dir = getattr(self.params, split)

            if not split_dir:
                continue

            images_dir = os.path.abspath(os.path.join(self.base_dir, split_dir))
            labels_dir = os.path.abspath(
                os.path.join(self.base_dir, split_dir, "../labels")
            )

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                raise FileNotFoundError(
                    f"Directory for images or labels not found at ({images_dir}) or ({labels_dir})"
                )

            for image_file in os.listdir(images_dir):
                image_path = os.path.abspath(os.path.join(images_dir, image_file))
                label_file = os.path.splitext(image_file)[0] + ".txt"
                label_path = os.path.abspath(os.path.join(labels_dir, label_file))
                if os.path.exists(label_path):
                    labeled_images.append(
                        LabeledImageRef(image_path=image_path, label_path=label_path)
                    )

        return labeled_images

    def get_split(
        self, proportion: float = 0.8, random_state: int = 42
    ) -> Tuple[List[LabeledImageRef], List[LabeledImageRef]]:
        if 0 > proportion or proportion > 1:
            raise ValueError(
                f"The proportion must be between 0 and 1, not {proportion}."
            )

        random.seed(random_state)

        labeled_images_copy = self.labeled_images.copy()

        random.shuffle(labeled_images_copy)

        split_idx = int(len(labeled_images_copy) * proportion)

        train_split = labeled_images_copy[:split_idx]
        val_split = labeled_images_copy[split_idx:]

        return train_split, val_split

    # TODO: improve this when the input for the models are clear


if __name__ == "__main__":
    ppl = FilePipeline(
        "C:/Users/Pedro/Downloads/Proyecto ML-20250129T041304Z-001/Proyecto ML/Imágenes/dataset/data.yaml"
    )

    train, valid = ppl.get_split()

    print(len(train), len(valid))
