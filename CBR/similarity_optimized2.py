import json
import os
from tqdm import tqdm
import cv2
import yaml
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from typing import Counter
from bboxes import load_labels
from spore_feature_extraction import extract_features
import similarity

def load_database(json_path):
    """Carga la base de datos de esporas desde un JSON."""
    with open(json_path, "r") as file:
        return json.load(file)

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

def calculate_dynamic_thresholds(database):
    """Calcula dos umbrales basados en los percentiles 80 y 95 de las similitudes previas."""
    similarity_scores = []
    values = list(database.values())

    for i in tqdm(range(len(values))):
        for j in range(i + 1, len(values)):
            similarity_scores.append(compare_cases(values[i], values[j]))

    return np.percentile(similarity_scores, [80, 95])

def find_similar_cases(new_case, database, thresholds, top_n=5):
    """Clasifica un nuevo caso según tres zonas de confianza y maneja aprendizaje activo."""
    similarities = []
    
    for espora_id, case in database.items():
        score = compare_cases(new_case, case)
        similarities.append((case["bounding_box"]["class"], score))

    # Ordenar por menor distancia (más similar)
    similarities.sort(key=lambda x: x[1])
    top_matches = similarities[:top_n]

    if not top_matches:
        return {"status": "review", "class": None, "top_matches": []}

    min_distance = top_matches[0][1]
    low_threshold, high_threshold = thresholds

    # Determinar la zona de confianza
    if min_distance <= low_threshold:
        # Zona de alta confianza: clasificación automática
        most_common = Counter([cls for cls, _ in top_matches]).most_common(1)[0][0]
        update_database(new_case, database, most_common)
        return {
            "status": "high_confidence",
            "class": most_common,
            "confidence": "auto",
            "top_matches": top_matches
        }
    elif min_distance <= high_threshold:
        # Zona de incertidumbre: modelo secundario
        secondary_class = similarity.predict(new_case, database)
        return {
            "status": "uncertainty",
            "class": secondary_class,
            "confidence": "secondary",
            "top_matches": top_matches
        }
    else:
        # Zona de revisión: aprendizaje activo
        system_recommendation = Counter([cls for cls, _ in top_matches]).most_common(1)[0][0]
        user_class = active_learning_prompt(system_recommendation, top_matches)
        if user_class is None:
            return {
                "status": "review",
                "class": None,
                "confidence": "manual",
                "top_matches": top_matches
            }
        else:
            return {
                "status": "high_confidence",
                "class": user_class,
                "confidence": "manual",
                "top_matches": top_matches
            }

def update_database(database, new_case, user_feedback):
    """Actualiza la base de datos con nuevos casos verificados (aprendizaje activo)."""
    new_id = str(int(max(database.keys(), key=lambda x: int(x))) + 1)
    new_case["bounding_box"]["class"] = user_feedback
    database[new_id] = new_case
    return database

def active_learning_prompt(system_recommendation, top_matches):
    """Interfaz para que los expertos validen casos difíciles."""
    print(f"Revisión requerida. Recomendación del sistema: {system_recommendation}")
    print("Casos similares de referencia:")
    for i, (cls, score) in enumerate(top_matches[:3], 1):
        print(f"{i}. Clase: {cls} - Distancia: {score:.2f}")
    
    while True:
        user_input = input("Introduzca la clase correcta (o 'skip' para omitir): ").strip()
        if user_input.lower() == 'skip':
            return None
        if user_input in {cls for cls, _ in top_matches}:
            return user_input
        print("Clase no válida. Las clases existentes son:", {cls for cls, _ in top_matches})

def predict(image, case_database, bounding_boxes):
    """ Predice el tipo de espora en una imagen usando CBR y aprendizaje automático. """
    
    results = []
    best_case = []

    for box in bounding_boxes:
        # 2. Extraer características de la espora detectada
        c, x, y, w, h = box
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
        best_case.append(find_similar_cases(all_features, case_database, (50.45, 63.08)))

      

    return best_case

case_database = load_database("../spore_features.json")

valid_image_folder = "../Imágenes/dataset/valid/images"
valid_image_files = {os.path.splitext(f)[0]: os.path.join(valid_image_folder, f) for f in os.listdir(valid_image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))}

valid_label_folder = "../Imágenes/dataset/valid/labels"
valid_label_files = {os.path.splitext(f)[0]: os.path.join(valid_label_folder, f) for f in os.listdir(valid_label_folder) if f.endswith('.txt')}

common_files = valid_image_files.keys() & valid_label_files.keys()

# Cargar el archivo YAML
with open("/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/Imágenes/dataset/data.yaml", "r") as file:
    data = yaml.safe_load(file)  # Carga el contenido del YAML

# Extraer la lista de nombres de las clases
class_names = data.get("names", [])  # Si "names" no existe, devuelve una lista vacía

for file_name in tqdm(common_files):
    image_path = valid_image_files[file_name]
    label_path = valid_label_files[file_name]

    image = cv2.imread(image_path)
    bboxes = load_labels(label_path, image.shape)

    resultados = predict(image, case_database, bboxes)

#     # Mostrar resultados
    i = 0
    for res in resultados:
        
        if not res['class'] == class_names[bboxes[i][0]]:
            print(f"Tipo detectado: {res['class']} Tipo real: {class_names[bboxes[i][0]]} ")
        i += 1
