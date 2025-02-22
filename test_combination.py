from CBR.classify_images import find_similar_cases
from CBR.database_utils import load_database, update_database
from CBR.image_features import extract_features
from CBR.image_segmentation import segment_image
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from time import time
from enum import Enum

from src.image_pipeline import find_bounding_boxes


class Combination(Enum):
    COMBO1 = 1
    COMBO2 = 2
    COMBO3 = 3


class PipelineTester:
    def __init__(self, combination: Combination):
        self.combination = combination
        self.metrics = {
            "total_time": 0,
            "segmentation_time": [],
            "classification_metrics": {},
            "active_learning_requests": 0,
        }

        # Prepare test data
        _, self.test_data = self.file_pipeline.get_split(proportion=0.2)

    def prepare_test_dataset(self):
        # Convert file references to image-label pairs
        test_dataset = []
        for img_ref in self.test_data:
            # Read image
            img = cv2.imread(img_ref.image_path)

            # Read label (assuming YOLO format)
            with open(img_ref.label_path, "r") as f:
                label_info = f.read().strip().split()
                label = int(label_info[0]) if label_info else -1

            test_dataset.append((img, label))

        return test_dataset

    # --- Segmentación ---
    def segment_image(self, image):
        start = time()
        if self.combination == Combination.COMBO1:
            # Use image_pipeline segmentation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = segment_image(gray)
            bboxes = find_bounding_boxes(thresh)

        elif self.combination == Combination.COMBO2:
            # Implement U-Net (requires pre-trained model)
            from unet_model import load_unet_model

            model = load_unet_model()
            mask = model.predict(image[np.newaxis, ...])[0]
            bboxes = self.postprocess_mask(mask)

        elif self.combination == Combination.COMBO3:
            # Implement Mask R-CNN
            from mrcnn import MaskRCNN

            model = MaskRCNN()
            results = model.detect([image], verbose=0)
            bboxes = results[0]["rois"]

        self.metrics["segmentation_time"].append(time() - start)
        return bboxes

    # --- Extracción de Características ---
    def extract_features(self, roi):
        if self.combination == Combination.COMBO1:
            # Implementación original
            return extract_features(roi)

        elif self.combination == Combination.COMBO2:
            # Autoencoder + estadísticas
            from autoencoder import encode_features

            stats = self._basic_statistics(roi)
            encoded = encode_features(roi)
            return {**stats, "encoded": encoded}

        elif self.combination == Combination.COMBO3:
            # SIFT + LBP
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(roi, None)
            lbp = local_binary_pattern(roi, P=8, R=1)
            return {"sift": des, "lbp": lbp}

    # --- Clasificación/Similitud ---
    def classify(self, features, database):
        if self.combination == Combination.COMBO1:
            # k-NN ponderado original
            return find_similar_cases(features, database)

        elif self.combination == Combination.COMBO2:
            # SVM con RBF kernel
            from sklearn.svm import SVC

            clf = SVC(kernel="rbf")
            return clf.predict(features)

        elif self.combination == Combination.COMBO3:
            # Random Forest
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier()
            return clf.predict(features)

    # --- Gestión de Conocimiento ---
    def update_knowledge(self, database, new_case):
        if self.combination in [Combination.COMBO1, Combination.COMBO3]:
            # Active Learning original
            return update_database(database, new_case)

        elif self.combination == Combination.COMBO2:
            # K-Means clustering
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=5)
            features = np.array([v["features"] for v in database.values()])
            kmeans.fit(features)
            return self._cluster_based_update(database, new_case, kmeans)

    # --- Métricas y Evaluación ---
    def calculate_stats(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        self.metrics["classification_metrics"] = {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
            "confusion_matrix": cm.tolist(),
        }
        return self.metrics

    def run_test_suite(self, database):
        # Prepare test dataset
        test_dataset = self.prepare_test_dataset()

        total_start = time()
        y_true, y_pred = [], []

        for img, label in test_dataset:
            bboxes = self.segment_image(img)
            for box in bboxes:
                x, y, w, h = box
                roi = img[y : y + h, x : x + w]
                features = self.extract_features(roi)
                pred = self.classify(features, database)
                y_pred.append(pred)
                y_true.append(label)

                if pred.get("status") == "review":
                    self.metrics["active_learning_requests"] += 1

                database = self.update_knowledge(database, features)

        self.metrics["total_time"] = time() - total_start
        return self.calculate_stats(y_true, y_pred)


# Main execution
if __name__ == "__main__":
    YAML_PATH = "path/to/your/data.yaml"

    database = load_database("spore_features.json")

    # Test different combinations
    tester_combo1 = PipelineTester(Combination.COMBO1, YAML_PATH)
    metrics1 = tester_combo1.run_test_suite(database)

    tester_combo2 = PipelineTester(Combination.COMBO2, YAML_PATH)
    metrics2 = tester_combo2.run_test_suite(database)

    tester_combo3 = PipelineTester(Combination.COMBO3, YAML_PATH)
    metrics3 = tester_combo3.run_test_suite(database)

    # Generate comparison report
    def generate_comparison_report(*metrics):
        report = {}
        for i, m in enumerate(metrics, 1):
            report[f"Combo{i}"] = {
                "Tiempo Total (s)": m["total_time"],
                "Precisión": m["classification_metrics"]["accuracy"],
                "F1-Score": m["classification_metrics"]["f1"],
                "Intervenciones Humanas": m["active_learning_requests"],
                "Tiempo Segmentación Promedio (s)": np.mean(m["segmentation_time"]),
            }
        return pd.DataFrame(report).T

    report = generate_comparison_report(metrics1, metrics2, metrics3)
    print(report)
