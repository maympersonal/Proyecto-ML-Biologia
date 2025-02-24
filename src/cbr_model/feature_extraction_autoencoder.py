import cv2
import numpy as np
from keras.models import Model


def autoencoder_image_to_vector(image: np.ndarray, model: Model) -> np.ndarray:
    """
    Extrae un vector de características combinando la representación latente
    obtenida a partir de un autoencoder y estadísticas globales de la imagen.

    Parámetros:
      - image: np.ndarray de tamaño 224x224 (imagen en formato BGR o RGB).
      - model: Modelo de autoencoder que retorna la representación latente al aplicar predict.

    Retorna:
      - Vector de características aplanado, resultante de la concatenación del vector latente
        y un vector con [media, desviación estándar, mediana] de la imagen.
    """
    if image.shape[:2] != (224, 224):
        raise ValueError("La imagen debe ser de tamaño 224x224")

    # Normalización de la imagen para el autoencoder
    image_normalized = image.astype("float32") / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    # Obtención de la representación latente
    latent_vector = model.predict(image_batch)
    latent_vector = latent_vector.flatten()

    # Cálculo de estadísticas de la imagen original (sin normalizar)
    mean_val = np.mean(image)
    std_val = np.std(image)
    median_val = np.median(image)

    stats = np.array([mean_val, std_val, median_val], dtype=np.float32)

    # Concatenar la representación latente con las estadísticas
    combined_vector = np.concatenate([latent_vector, stats])
    return combined_vector
