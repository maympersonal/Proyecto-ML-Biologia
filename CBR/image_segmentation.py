import cv2
import numpy as np
from tensorflow.keras.models import load_model


def segment_image(image):
    """Segmenta la imagen para detectar esporas."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    return bounding_boxes


def segment_image_unet(image):
    """Segmenta la imagen utilizando un modelo U-Net preentrenado para detectar esporas."""
    # Cargar modelo U-Net preentrenado (se recomienda cargarlo una sola vez en producción)
    model = load_model("unet_model.h5")

    orig_h, orig_w = image.shape[:2]
    # Convertir la imagen a escala de grises (si el modelo fue entrenado en 1 canal)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Redimensionar la imagen al tamaño que espera el modelo, por ejemplo, 256x256
    resized = cv2.resize(gray, (256, 256))
    # Normalización y ajuste de dimensiones: (1, 256, 256, 1)
    input_img = resized.astype("float32") / 255.0
    input_img = np.expand_dims(input_img, axis=-1)
    input_img = np.expand_dims(input_img, axis=0)

    # Predecir la máscara de segmentación
    pred_mask = model.predict(input_img)[0, ..., 0]
    # Umbralizar la máscara (valor de 0.5 como ejemplo)
    mask = (pred_mask > 0.5).astype(np.uint8) * 255
    # Redimensionar la máscara a las dimensiones originales
    mask = cv2.resize(mask, (orig_w, orig_h))

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    return bounding_boxes


def segment_image_mask_rcnn(image):
    """Segmenta la imagen utilizando un modelo Mask R-CNN preentrenado para detectar esporas."""
    # Cargar el modelo Mask R-CNN. Se asume la existencia de 'frozen_inference_graph.pb'
    # y 'mask_rcnn_config.pbtxt' en el directorio actual.
    net = cv2.dnn.readNetFromTensorflow(
        "frozen_inference_graph.pb", "mask_rcnn_config.pbtxt"
    )

    # Crear un blob a partir de la imagen
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)

    # Realizar la detección. Los nombres de salida pueden variar según el modelo.
    detections, masks = net.forward(["detection_out_final", "detection_masks"])

    h, w = image.shape[:2]
    bounding_boxes = []
    for i in range(detections.shape[2]):
        score = detections[0, 0, i, 2]
        if score > 0.5:  # Filtrar detecciones con baja confianza
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bounding_boxes.append((x1, y1, x2 - x1, y2 - y1))
    return bounding_boxes
