import cv2
import numpy as np
import os
from tqdm import tqdm

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

def segment_image(image):
    """ Segmenta la imagen para detectar esporas. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    
    return bounding_boxes

def draw_bounding_box(image_path, bounding_boxes):
    """
    Muestra la imagen con los bounding boxes dibujados en rojo.

    :param image_path: Ruta de la imagen.
    :param bounding_boxes: Lista de bounding boxes en formato (x, y, width, height).
                           Se asume que (x, y) es la esquina superior izquierda.
    """
    # Cargar la imagen
    image = cv2.imread(image_path)

    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return
    
    # Dibujar los bounding boxes
    for (x, y, w, h) in bounding_boxes:
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)  # Rojo, grosor 2

    # Mostrar la imagen con los bounding boxes
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_and_compare_bounding_boxes(image_path, bounding_boxes1, bounding_boxes2):
    """
    Muestra la imagen dos veces, una con bounding_boxes1 y otra con bounding_boxes2, 
    colocándolas una al lado de la otra para facilitar la comparación.

    :param image_path: Ruta de la imagen.
    :param bounding_boxes1: Lista de bounding boxes para la primera imagen.
    :param bounding_boxes2: Lista de bounding boxes para la segunda imagen.
    """
    # Cargar la imagen
    image = cv2.imread(image_path)

    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Crear copias de la imagen
    img1 = image.copy()
    img2 = image.copy()

    # Dibujar los bounding boxes en la primera copia
    for (x, y, w, h) in bounding_boxes1:
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))
        cv2.rectangle(img1, top_left, bottom_right, (0, 0, 255), 2)  # Rojo

    # Dibujar los bounding boxes en la segunda copia
    for (c, x, y, w, h) in bounding_boxes2:
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))
        cv2.rectangle(img2, top_left, bottom_right, (255, 0, 0), 2)  # Azul

    # Concatenar las imágenes horizontalmente
    combined_image = np.hstack((img1, img2))

    # Mostrar la imagen combinada
    cv2.imshow("Bounding Box Comparison", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    valid_image_folder = "../Imágenes/dataset/valid/images"
    valid_image_files = {os.path.splitext(f)[0]: os.path.join(valid_image_folder, f) for f in os.listdir(valid_image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))}

    valid_label_folder = "../Imágenes/dataset/valid/labels"
    valid_label_files = {os.path.splitext(f)[0]: os.path.join(valid_label_folder, f) for f in os.listdir(valid_label_folder) if f.endswith('.txt')}

    common_files = valid_image_files.keys() & valid_label_files.keys()

    for file_name in tqdm(common_files):
        image_path = valid_image_files[file_name]
        label_path = valid_label_files[file_name]

        image = cv2.imread(image_path)
        bboxes = load_labels(label_path, image.shape)


        # draw_bounding_box(image_path, segment_image(image))
        draw_and_compare_bounding_boxes(image_path, segment_image(image), bboxes)

# main()
