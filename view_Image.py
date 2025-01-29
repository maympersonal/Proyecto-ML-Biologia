import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters, morphology

# Rutas de las carpetas
imagenes_folder = "/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/dataset-20250125T205521Z-001/dataset/train/images/"  # Cambia esto por la ruta de tus imágenes
labels_folder = "/media/daniman/Dani/UNI/4toaño/Machine Learning/Proyecto/dataset-20250125T205521Z-001/dataset/train/labels/"      # Cambia esto por la ruta de tus labels

# Obtener la lista de archivos en la carpeta de imágenes
imagenes = [f for f in os.listdir(imagenes_folder) if f.endswith(".jpg")]  # Cambia la extensión si es necesario

# Definir el margen en pixeles
margin = 1

# Recorrer cada imagen
for imagen in imagenes:

    # Obtener el nombre del archivo sin extensión
    nombre_sin_extension = os.path.splitext(imagen)[0]

    # Ruta de la imagen
    image_path = imagenes_folder + nombre_sin_extension + ".jpg"

    # Cargar la imagen
    image = cv2.imread(image_path)

    # Convertir la imagen de BGR (OpenCV) a RGB (Matplotlib)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Definir las dimensiones de la imagen
    height, width, _ = image.shape

    # Crear una máscara global vacía (todo en 0)
    global_mask = np.ones((height, width), dtype=np.uint8)

    # Ruta del archivo de label correspondiente
    label_path = os.path.join(labels_folder, nombre_sin_extension + ".txt")

    # Verificar si el archivo de label existe
    if os.path.exists(label_path):
        # Leer el contenido del archivo de label
        with open(label_path, "r") as file:
            lines = file.readlines()

        # Procesar cada línea del archivo de label
        for line in lines:
            # Eliminar espacios en blanco y dividir la línea en sus componentes
            label_data = line.strip().split()

            # Extraer los valores del label
            class_id = int(label_data[0])  # ID de la clase
            x_center = float(label_data[1])  # Coordenada x del centro (normalizada)
            y_center = float(label_data[2])  # Coordenada y del centro (normalizada)
            bbox_width = float(label_data[3])    # Ancho del bounding box (normalizado)
            bbox_height = float(label_data[4])   # Alto del bounding box (normalizado)

            # Convertir coordenadas normalizadas a píxeles
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            # Calcular las coordenadas del bounding box (x_min, y_min, x_max, y_max)
            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)

            # Aplicar un margen a las coordenadas del bounding box
            x_min = max(0, x_min - margin)  # Asegurarse de no salirse de los límites de la imagen
            y_min = max(0, y_min - margin)
            x_max = min(width, x_max + margin)
            y_max = min(height, y_max + margin)

            # Recortar la región de interés (ROI)
            roi = image[y_min:y_max, x_min:x_max]

            # Convertir la ROI a escala de grises
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

            # Aplicar un umbral para crear una máscara binaria
            _, mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            inv_mask = cv2.bitwise_not(mask)

            # Añadir la máscara de la ROI a la máscara global
            global_mask[y_min:y_max, x_min:x_max] = cv2.bitwise_or(global_mask[y_min:y_max, x_min:x_max], inv_mask)

            # Dibujar el bounding box en la imagen
            # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Color azul, grosor 2

            # Mostrar la ROI y la máscara
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(roi)
            plt.title("Región de Interés (ROI)")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Máscara Binaria Umbral=127")
            plt.axis("off")

            plt.show()

        # Invertir la máscara global
        global_mask_inv = cv2.bitwise_not(global_mask)

        # Mostrar la imagen original y la máscara global
        plt.figure(figsize=(15, 5))

        # Imagen original con bounding boxes
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Imagen Original con Bounding Boxes")
        plt.axis("off")

        # Máscara global
        plt.subplot(1, 2, 2)
        plt.imshow(global_mask_inv, cmap="gray")
        plt.title("Máscara Binaria Global")
        plt.axis("off")

        plt.show()

