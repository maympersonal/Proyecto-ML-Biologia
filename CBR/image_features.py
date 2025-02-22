import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_features(image):
    """Extrae características de color, textura y estadísticas de una imagen."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Estadísticas básicas
    mean_gray = float(np.mean(gray))
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
    list_rgb = [
        cv2.calcHist([image], [i], None, [256], [0, 256]).flatten().tolist()
        for i in range(3)
    ]
    hist_rgb = [
        list([float(n) for n in h]) for h in list_rgb
    ]  # Cantidad de píxeles para cada posible intensidad de color (de 0 a 255).

    # Histograma de color en HSV (normalizado)
    list_hsv = [
        cv2.calcHist([hsv], [i], None, [256], [0, 256]).flatten() for i in range(3)
    ]
    hist_hsv = [h / h.sum() for h in list_hsv]  # Normalización

    # Textura: características GLCM
    glcm = graycomatrix(
        gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    contrast = graycoprops(glcm, "contrast")[0, 0]
    dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]

    # Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum() + 1e-7  # Normalización

    return {
        "color_features": {
            "mean_hsv": mean_hsv,
            "std_hsv": std_hsv,
            "hist_hsv": [h.tolist() for h in hist_hsv],  # Convertir a lista para JSON
        },
        "texture_features": {
            "contrast": contrast,
            "dissimilarity": dissimilarity,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation,
            "lbp_histogram": lbp_hist.tolist(),
        },
        "stats": {
            "mean_gray": mean_gray,
            "std_gray": std_gray,
            "min_gray": min_gray,
            "max_gray": max_gray,
            "hu_moments": hu_moments,
        },
    }
