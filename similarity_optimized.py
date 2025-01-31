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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


def extract_feature_vector(case):
    """Extrae todas las características de un caso en un vector único."""
    # Características del bounding box
    bbox = [case["bounding_box"]["width"], case["bounding_box"]["height"]]
    
    # Estadísticas de intensidad
    stats = [
        case["features"]["stats"]["mean_gray"],
        case["features"]["stats"]["std_gray"],
        case["features"]["stats"]["min_gray"],
        case["features"]["stats"]["max_gray"]
    ]
    
    # Histograma de color concatenado
    color_hist = np.concatenate(case["features"]["color_features"]["hist_hsv"])
    
    # Características de textura
    texture = [
        case["features"]["texture_features"]["contrast"],
        case["features"]["texture_features"]["homogeneity"],
        case["features"]["texture_features"]["energy"],
        case["features"]["texture_features"]["correlation"]
    ]
    
    # Momentos de Hu
    hu_moments = case["features"]["stats"]["hu_moments"]
    
    # Histograma LBP
    lbp_hist = case["features"]["texture_features"]["lbp_histogram"]
    
    # Combinar todas las características en un solo vector
    return np.concatenate([
        np.array(bbox),
        np.array(stats),
        color_hist,
        np.array(texture),
        np.array(hu_moments),
        np.array(lbp_hist)
    ])

def find_similar_cases(new_case, database, threshold=70, top_n=5):
    """Encuentra los casos más similares con KNN ponderado y desempate logístico."""
    # Prepara datos para posible clasificación logística
    X_train = []
    y_train = []
    distances = []
    
    # Extraer características del nuevo caso
    new_case_vector = extract_feature_vector(new_case)
    
    for espora_id, case in database.items():
        # Calcular distancia
        distance = compare_cases(new_case, case)
        distances.append((case["bounding_box"]["class"], distance))
        
        # Almacenar características para entrenamiento
        case_vector = extract_feature_vector(case)
        X_train.append(case_vector)
        y_train.append(case["bounding_box"]["class"])
    
    # Ordenar por menor distancia
    distances.sort(key=lambda x: x[1])
    top_distances = distances[:top_n]
    
    # Calcular pesos por clase (inverso de la distancia)
    class_weights = {}
    for class_label, distance in top_distances:
        weight = 1 / (distance + 1e-7)  # Evitar división por cero
        class_weights[class_label] = class_weights.get(class_label, 0.0) + weight
    
    # Encontrar la clase con mayor peso acumulado
    max_weight = max(class_weights.values())
    candidates = [cls for cls, w in class_weights.items() if w == max_weight]
    
    if len(candidates) == 1:
        predicted_class = candidates[0]
    else:
        # Desempate con regresión logística
        try:
            # Escalar características
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            new_case_scaled = scaler.transform([new_case_vector])
            
            # Entrenar y predecir
            clf = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
            clf.fit(X_scaled, y_train)
            predicted_class = clf.predict(new_case_scaled)[0]
        except:
            predicted_class = "Clasificación manual requerida"
    
    # Verificar umbral de distancia mínima
    min_distance = top_distances[0][1]
    if min_distance > threshold:
        return "Clasificación manual requerida"
    
    return predicted_class

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
        best_case.append(find_similar_cases(all_features, case_database))

      

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
        
        if not res == class_names[bboxes[i][0]]:
            print(res)
            print(f"Tipo detectado: {res} Tipo real: {class_names[bboxes[i][0]]} ")
        # print(f"Detectado satisfactoriamente: {res[0][0][0] == class_names[bboxes[i][0]]}")
        i += 1
