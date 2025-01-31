import os
import cv2
import yaml
import json
import numpy as np
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def load_labels(label_path, image_shape):
    """Carga los bounding boxes desde un archivo de etiquetas YOLO."""
    h, w = image_shape[:2]
    bboxes = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        x_center, y_center, width, height = map(float, values[1:])

        # Convertir coordenadas normalizadas a píxeles
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        box_width = int(width * w)
        box_height = int(height * h)

        bboxes.append((class_id, x_min, y_min, box_width, box_height))

    return bboxes

def extract_features(image):
    """Extrae características de color, textura y estadísticas de una imagen."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Estadísticas básicas
    mean_gray =float(np.mean(gray))
    std_gray = float(np.std(gray))
    min_gray = float(np.min(gray))
    max_gray = float(np.max(gray))

    # Características de forma (Momentos de Hu)
    moments = cv2.moments(gray)
    hu_moments = list(cv2.HuMoments(moments).flatten())

    # Características de color (media y desviación estándar en RGB)
    mean_rgb = list(np.mean(image, axis=(0, 1)).tolist())
    std_rgb = list(np.std(image, axis=(0, 1)).tolist())

    # Características de color (media y desviación estándar en HSV)
    mean_hsv = list(np.mean(hsv, axis=(0, 1)).tolist())
    std_hsv = list(np.std(hsv, axis=(0, 1)).tolist())

    # Histograma de color en RGB
    list_rgb = [cv2.calcHist([image], [i], None, [256], [0, 256]).flatten().tolist() for i in range(3)]
    hist_rgb = [list([float(n) for n in h]) for h in list_rgb] # Cantidad de píxeles para cada posible intensidad de color (de 0 a 255).

    # Histograma de color en HSV (normalizado)
    list_hsv = [cv2.calcHist([hsv], [i], None, [256], [0, 256]).flatten() for i in range(3)]
    hist_hsv = [h / h.sum() for h in list_hsv]  # Normalización

    # Textura: características GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalización

    return {
        "color_features": {
            "mean_hsv": mean_hsv,
            "std_hsv": std_hsv,
            "hist_hsv": [h.tolist() for h in hist_hsv]  # Convertir a lista para JSON
        },
        "texture_features": {
            "contrast": contrast,
            "dissimilarity": dissimilarity,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation,
            "lbp_histogram": lbp_hist.tolist()
        },
        "stats": {
            "mean_gray": mean_gray,
            "std_gray": std_gray,
            "min_gray": min_gray,
            "max_gray": max_gray,
            "hu_moments": hu_moments
        }
    }

def process_images(image_folder, label_folder, output_json):
    """Procesa todas las imágenes en la carpeta y extrae características de cada espora."""
    all_features = {}

    # Obtener lista de archivos en ambas carpetas
    image_files = {os.path.splitext(f)[0]: os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))}
    label_files = {os.path.splitext(f)[0]: os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.txt')}

    # Procesar solo los archivos que tienen imagen y label correspondiente
    common_files = image_files.keys() & label_files.keys()

    # Cargar el archivo YAML
    with open("/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/Imágenes/dataset/data.yaml", "r") as file:
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

# Ejemplo de uso
image_folder = "/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/Imágenes/dataset/train/images/"
label_folder = "/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/Imágenes/dataset/train/labels/"
output_json = "/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/spore_features.json"

# process_images(image_folder, label_folder, output_json)

