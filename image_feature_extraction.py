import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray

def extract_global_features(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. Características de color
    color_features = {}
    color_features['mean_rgb'] = list(np.mean(image, axis=(0, 1)))  # Promedio por canal (R, G, B)
    color_features['std_rgb'] = list(np.std(image, axis=(0, 1)))    # Desviación estándar por canal
    hist_rgb = [cv2.calcHist([image], [i], None, [256], [0, 256]).flatten() for i in range(3)]
    color_features['hist_rgb'] = [list([float(n) for n in h]) for h in hist_rgb] # Cantidad de píxeles para cada posible intensidad de color (de 0 a 255).
    
    # 2. Características de textura (GLCM)
    glcm = graycomatrix(gray, [1], [0, float(np.pi/4), float(np.pi/2), float(3*np.pi/4)], levels=256, symmetric=True, normed=True)
    texture_features = {}
    texture_features['contrast'] = graycoprops(glcm, 'contrast').mean() # Diferencia entre los valores de gris de los píxeles vecinos.
    texture_features['dissimilarity'] = graycoprops(glcm, 'dissimilarity').mean() # Igual pero con menor peso en diferencias pequeñas.
    texture_features['homogeneity'] = graycoprops(glcm, 'homogeneity').mean() # Cuán uniformes son los valores de gris en la imagen.
    texture_features['energy'] = graycoprops(glcm, 'energy').mean() # Uniformidad de la imagen.
    texture_features['correlation'] = graycoprops(glcm, 'correlation').mean() # Relación entre los valores de gris de los píxeles vecinos.
    
    # 3. Características de textura (LBP)
    # Frecuencia de cada patrón de textura en la imagen.
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum()  # Normalizar
    texture_features['lbp_histogram'] = list(lbp_hist)
    
    # 4. Estadísticas básicas
    stats = {}
    stats['mean_gray'] = float(np.mean(gray)) # Promedio de los valores de los píxeles en escala de grises.
    stats['std_gray'] = float(np.std(gray)) # Desviación estándar de los valores en escala de grises.
    stats['min_gray'] = float(np.min(gray)) # Valor mínimo de gris en la imagen 
    stats['max_gray'] = float(np.max(gray)) # Valor máximo de gris en la imagen
    
    # Combinar todas las características
    features = {
        'color_features': color_features,
        'texture_features': texture_features,
        'stats': stats
    }
    
    return features

def save_features_to_json_multiple(image_features, output_path):
    """
    Guarda características de múltiples imágenes en un archivo JSON.

    Args:
        image_features (dict): Diccionario con las características de múltiples imágenes.
        output_path (str): Ruta del archivo JSON de salida.
    """

    # Guardar en el archivo JSON
    with open(output_path, 'w') as json_file:
        json.dump(image_features, json_file, indent=4)

def process_images_and_save(image_folder, output_path):
    """
    Procesa un grupo de imágenes, extrae características y las guarda en un archivo JSON.

    Args:
        image_folder (str): Carpeta que contiene las imágenes.
        output_path (str): Ruta del archivo JSON de salida.
    """
    all_features = {}

    for image_name in tqdm(os.listdir(image_folder), desc="Loading images"):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(image_folder, image_name)
            features = extract_global_features(image_path)  # Usamos la función previa
            all_features[image_name] = features  # Guardar con el nombre de la imagen como clave

    # Guardar todas las características en un archivo JSON
    print(all_features)
    save_features_to_json_multiple(all_features, output_path)

image_folder = "/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/Imágenes/dataset/train/images/"
output_path = "/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/features.json"
process_images_and_save(image_folder, output_path)

