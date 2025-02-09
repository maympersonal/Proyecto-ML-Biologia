import json
import numpy as np
import cv2
import os
import yaml
from tqdm import tqdm
from database_utils import load_labels, update_database
from image_features import extract_features
from image_segmentation import segment_image
from classify_images import find_similar_cases
class CBR_model:
    def __init__(self,k = 3):
        self.k = k
        pass
    
    def fit(self, image_folder, label_folder, output_json):
        """Procesa todas las imágenes en la carpeta y extrae características de cada espora."""
        all_features = {}

        # Obtener lista de archivos en ambas carpetas
        image_files = {os.path.splitext(f)[0]: os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))}
        label_files = {os.path.splitext(f)[0]: os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.txt')}

        # Procesar solo los archivos que tienen imagen y label correspondiente
        common_files = image_files.keys() & label_files.keys()

        # Cargar el archivo YAML
        try:
            with open(".\\Imagenes\\dataset\\data.yaml", "r") as file:
                data = yaml.safe_load(file)  # Carga el contenido del YAML
        except:
            with open("../Imagenes/dataset/data.yaml", "r") as file:
                data = yaml.safe_load(file)  # Carga el contenido del YAML

        # Extraer la lista de nombres de las clases
        class_names = data.get("names", [])  # Si "names" no existe, devuelve una lista vacía

        for file_name in tqdm(common_files):
            image_path = image_files[file_name]
            label_path = label_files[file_name]

            # print(f"Procesando: {image_path} con {label_path}")

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                continue

            bboxes = load_labels(label_path, image.shape)

            for i, (class_id, x_min, y_min, width, height) in enumerate(bboxes):
                roi = image[y_min:y_min+height, x_min:x_min+width]

                if roi.size == 0:
                    continue

                features = extract_features(roi)
                # Agregar información del bounding box
                espora_id = f"{file_name}_espora_{i}_class_{class_id}"
                all_features[espora_id] = {
                    "bounding_box": {
                        "class": class_names[class_id],
                        "x_min": x_min,
                        "y_min": y_min,
                        "width": width,
                        "height": height
                    },
                    "features": features
                }

        # Guardar en JSON
        with open(output_json, "w") as json_file:
            json.dump(all_features, json_file, indent=4)
        return
    
    def predict(self,image,image_name,case_database, threshold):
        """ Predice el tipo de espora en una imagen usando CBR y aprendizaje automático. """
        # 1. Segmentar la imagen
        bounding_boxes = segment_image(image)
        best_case=[]
        for i,box in enumerate(bounding_boxes):
            # 2. Extraer características de la espora detectada
            image_name= f"{image_name}_espora_{i}"
            x, y, w, h = box
            roi = image[y:y+h, x:x+w]
            features = extract_features(roi)
            all_features = {
                    "bounding_box": {
                        "class": '',
                        "x_min": x,
                        "y_min": y,
                        "width": w,
                        "height": h
                    },
                    "features": features
                }

            # 3. Buscar el caso más similar en la base de datos
            best_case.append(find_similar_cases(all_features,image_name, case_database, threshold, self.k))

        return best_case
