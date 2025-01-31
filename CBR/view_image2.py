import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters, morphology

# Rutas de imágenes y etiquetas
image_dir = "/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/dataset-20250125T205521Z-001/dataset/train/images/"
label_dir = "/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/dataset-20250125T205521Z-001/dataset/train/labels/"

dataset = []

def create_precise_mask(image, bbox):
    """
    Segmenta la espora dentro del bounding box usando técnicas de procesamiento de imágenes.
    """
    x_min, y_min, x_max, y_max = bbox
    roi = image[y_min:y_max, x_min:x_max]
    
    # Convertir a escala de grises y normalizar
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # gray = exposure.equalize_hist(gray)
    
    # Ecualización del histograma con OpenCV (mantiene el tipo uint8)
    gray_eq = cv2.equalizeHist(gray)

    # Suavizado y detección de bordes
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Operaciones morfológicas para cerrar huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Encontrar contornos y seleccionar el más grande (suponiendo que es la espora)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask_roi = np.zeros_like(gray)
        cv2.drawContours(mask_roi, [largest_contour], -1, 255, -1)
        
        # Eliminar pequeños objetos (opcional)
        mask_roi = morphology.remove_small_objects(mask_roi.astype(bool), min_size=50).astype(np.uint8) * 255
        return mask_roi
    else:
        return np.zeros_like(gray)

def visualize_image_and_mask(image_path, mask):
    """
    Muestra la imagen original y su máscara lado a lado.
    
    Args:
        image_path (str): Ruta de la imagen original.
        mask (numpy.ndarray): Máscara binaria (2D array de 0s y 255s).
    """
    # Cargar imagen en RGB para matplotlib
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Mostrar imagen original
    axes[0].imshow(image)
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")
    
    # Mostrar máscara
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Máscara Generada")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

# Procesar todas las imágenes
for image_name in os.listdir(image_dir):
    if image_name.endswith(('.png', '.jpg')):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")
        
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            class_id, x_center, y_center, bbox_w, bbox_h = map(float, line.strip().split())
            
            # Convertir coordenadas YOLO a píxeles
            x_center = int(x_center * width)
            y_center = int(y_center * height)
            bbox_w = int(bbox_w * width)
            bbox_h = int(bbox_h * height)
            
            x_min = max(0, x_center - bbox_w // 2)
            y_min = max(0, y_center - bbox_h // 2)
            x_max = min(width, x_center + bbox_w // 2)
            y_max = min(height, y_center + bbox_h // 2)
            
            # Crear máscara precisa dentro del ROI
            precise_mask_roi = create_precise_mask(image, (x_min, y_min, x_max, y_max))
            
            # Insertar la máscara en la imagen completa
            mask[y_min:y_max, x_min:x_max] = precise_mask_roi
        
        dataset.append({
            'image_path': image_path,
            'mask': mask.tolist(),
            'features': {}
        })

        visualize_image_and_mask(image_path, mask)

# # Guardar el dataset
# with open('dataset_precise.json', 'w') as f:
#     json.dump(dataset, f)


