import json
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import cv2
import os
import yaml
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from typing import Counter



def chi_square_distance(hist1, hist2):
    """Calcula la distancia Chi-cuadrado entre dos histogramas."""
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-7))

def compare_cases(case1, case2):
    """Calcula la similitud entre dos esporas usando varias métricas."""
    
    # Comparación de bounding box (penaliza esporas con tamaños muy diferentes)
    bbox_diff = abs(case1["bounding_box"]["width"] - case2["bounding_box"]["width"]) + \
                abs(case1["bounding_box"]["height"] - case2["bounding_box"]["height"])

    # Comparación de estadísticas básicas (media, desviación estándar, valores mínimo y máximo)
    stats1 = np.array([case1["features"]["stats"]["mean_gray"], 
                        case1["features"]["stats"]["std_gray"], 
                        case1["features"]["stats"]["min_gray"], 
                        case1["features"]["stats"]["max_gray"]])
    stats2 = np.array([case2["features"]["stats"]["mean_gray"], 
                        case2["features"]["stats"]["std_gray"], 
                        case2["features"]["stats"]["min_gray"], 
                        case2["features"]["stats"]["max_gray"]])
    stats_distance = euclidean(stats1, stats2)

    # Comparación de histogramas de color en HSV
    hist1 = np.concatenate(case1["features"]["color_features"]["hist_hsv"])
    hist2 = np.concatenate(case2["features"]["color_features"]["hist_hsv"])
    hist_similarity = cosine(hist1, hist2)

    # Comparación de características de textura (GLCM)
    texture1 = np.array([
        case1["features"]["texture_features"]["contrast"],
        case1["features"]["texture_features"]["homogeneity"],
        case1["features"]["texture_features"]["energy"],
        case1["features"]["texture_features"]["correlation"]
    ])
    texture2 = np.array([
        case2["features"]["texture_features"]["contrast"],
        case2["features"]["texture_features"]["homogeneity"],
        case2["features"]["texture_features"]["energy"],
        case2["features"]["texture_features"]["correlation"]
    ])
    texture_distance = euclidean(texture1, texture2)

    # Comparación de momentos de Hu (medida de similitud basada en distancia euclidiana)
    hu1 = np.array(case1["features"]["stats"]["hu_moments"])
    hu2 = np.array(case2["features"]["stats"]["hu_moments"])
    hu_distance = euclidean(hu1, hu2)

    # Comparación de LBP (usando distancia de Chi-cuadrado)
    lbp1 = np.array(case1["features"]["texture_features"]["lbp_histogram"])
    lbp2 = np.array(case2["features"]["texture_features"]["lbp_histogram"])
    lbp_distance = chi_square_distance(lbp1, lbp2)

    # Ponderación de las similitudes
    similarity_score = (stats_distance * 0.2) + (hist_similarity * 0.25) + \
                       (texture_distance * 0.2) + (hu_distance * 0.2) + \
                       (lbp_distance * 0.1) + (bbox_diff * 0.05)

    return similarity_score

def find_similar_cases(new_case, database, top_n=5):
    """Encuentra los casos más similares en la base de datos."""
    similarities = []

    for espora_id, case in database.items():
        score = compare_cases(new_case, case)
        similarities.append((database[espora_id]["bounding_box"]["class"], score))

    # Ordenar por menor distancia (más similar)
    similarities.sort(key=lambda x: x[1])

    k_values = similarities[:top_n]
    threshold = 70
    # Decisión basada en umbral
    #print(min(k_values)[1])
    if min(k_values)[1] > threshold:
        return "Clasificación manual requerida"
    else:
        values = [tupla[0] for tupla in k_values]
        most_common = Counter(values).most_common()
        return most_common
    
def calculate_dynamic_threshold(database):
    """Calcula un umbral basado en el percentil 90 de las similitudes previas."""
    similarity_scores = []

    for i in range(len(database)):
        for j in range(i + 1, len(database)):
            similarity_scores.append(compare_cases(database[i], database[j]))

    return float(np.percentile(similarity_scores, 90))  # Usa el percentil 90 como umbral
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

def load_database(json_path):
    """Carga la base de datos de esporas desde un JSON."""
    try:
        with open(json_path, "r") as file:
            return json.load(file)
    except:
        try:
            image_folder = ".\\Imágenes\\dataset\\train\\images"
            label_folder = ".\\Imágenes\\dataset\\train\\labels"
            output_json = ".\\spore_features.json"
        except:
            image_folder = "../Imágenes/dataset/train/images"
            label_folder = "../Imágenes/dataset/train/labels"
            output_json = "../spore_features.json"
        process_images(image_folder, label_folder, output_json)
      
def segment_image(image):
    """ Segmenta la imagen para detectar esporas. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    return bounding_boxes

class CBR:
    def __init__(self,k = 3):
        self.k = k
        pass
    def fit(self, image_folder, label_folder, output_json):
        process_images(image_folder, label_folder, output_json)
        return
    def predict(self,image,case_database):
        """ Predice el tipo de espora en una imagen usando CBR y aprendizaje automático. """
        # 1. Segmentar la imagen
        bounding_boxes = segment_image(image)
        best_case=[]
        for box in bounding_boxes:
            # 2. Extraer características de la espora detectada
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
            best_case.append( find_similar_cases(all_features, case_database))

      

        return best_case
