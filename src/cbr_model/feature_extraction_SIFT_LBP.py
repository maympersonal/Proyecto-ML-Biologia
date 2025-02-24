import cv2
import numpy as np
from keras.models import Model


def compute_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Calcula el Local Binary Pattern (LBP) para una imagen en escala de grises.

    Parámetros:
      - gray: Imagen en escala de grises.

    Retorna:
      - Imagen LBP con valores codificados para cada píxel (se omiten los bordes).
    """
    h, w = gray.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    # Para cada píxel (excluyendo los bordes)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = gray[i, j]
            # Vecinos en orden: arriba-izquierda, arriba, arriba-derecha, derecha, abajo-derecha, abajo, abajo-izquierda, izquierda.
            neighbors = [
                gray[i - 1, j - 1],
                gray[i - 1, j],
                gray[i - 1, j + 1],
                gray[i, j + 1],
                gray[i + 1, j + 1],
                gray[i + 1, j],
                gray[i + 1, j - 1],
                gray[i, j - 1],
            ]
            lbp_value = 0
            for idx, neighbor in enumerate(neighbors):
                if neighbor >= center:
                    lbp_value |= 1 << idx
            lbp[i - 1, j - 1] = lbp_value
    return lbp


def sift_lbp_image_to_vector(image: np.ndarray, model: Model) -> np.ndarray:
    """
    Extrae un vector de características combinando SIFT y LBP:
      - SIFT: Se calcula la media de los descriptores SIFT (vector de 128 dimensiones).
      - LBP: Se obtiene el histograma normalizado (256 bins) de la imagen LBP.

    Parámetros:
      - image: np.ndarray de tamaño 224x224.
      - model: Este parámetro no se utiliza en esta implementación.

    Retorna:
      - Vector de características aplanado resultante de la concatenación del descriptor SIFT medio
        y el histograma de LBP.
    """
    if image.shape[:2] != (224, 224):
        raise ValueError("La imagen debe ser de tamaño 224x224")

    # Convertir la imagen a escala de grises (si no lo está)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Extracción de características SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) == 0:
        sift_vector = np.zeros(128, dtype=np.float32)
    else:
        sift_vector = np.mean(descriptors, axis=0)

    # Cálculo de LBP y su histograma
    lbp_image = compute_lbp(gray)
    hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-7  # Normalización para evitar división por cero

    # Concatenar las características SIFT y el histograma de LBP
    combined_vector = np.concatenate([sift_vector, hist])
    return combined_vector
