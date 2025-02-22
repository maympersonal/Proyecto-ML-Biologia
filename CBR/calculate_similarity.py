import numpy as np
from scipy.spatial.distance import euclidean, cosine
from tqdm import tqdm


def compare_cases(case1, case2):
    """Calcula la similitud entre dos esporas usando varias métricas."""

    # Comparación de bounding box (penaliza esporas con tamaños muy diferentes)
    bbox_diff = abs(
        case1["bounding_box"]["width"] - case2["bounding_box"]["width"]
    ) + abs(case1["bounding_box"]["height"] - case2["bounding_box"]["height"])

    # Comparación de estadísticas básicas (media, desviación estándar, valores mínimo y máximo)
    stats1 = np.array(
        [
            case1["features"]["stats"]["mean_gray"],
            case1["features"]["stats"]["std_gray"],
            case1["features"]["stats"]["min_gray"],
            case1["features"]["stats"]["max_gray"],
        ]
    )
    stats2 = np.array(
        [
            case2["features"]["stats"]["mean_gray"],
            case2["features"]["stats"]["std_gray"],
            case2["features"]["stats"]["min_gray"],
            case2["features"]["stats"]["max_gray"],
        ]
    )
    stats_distance = euclidean(stats1, stats2)

    # Comparación de histogramas de color en HSV
    hist1 = np.concatenate(case1["features"]["color_features"]["hist_hsv"])
    hist2 = np.concatenate(case2["features"]["color_features"]["hist_hsv"])
    hist_similarity = cosine(hist1, hist2)

    # Comparación de características de textura (GLCM)
    texture1 = np.array(
        [
            case1["features"]["texture_features"]["contrast"],
            case1["features"]["texture_features"]["homogeneity"],
            case1["features"]["texture_features"]["energy"],
            case1["features"]["texture_features"]["correlation"],
        ]
    )
    texture2 = np.array(
        [
            case2["features"]["texture_features"]["contrast"],
            case2["features"]["texture_features"]["homogeneity"],
            case2["features"]["texture_features"]["energy"],
            case2["features"]["texture_features"]["correlation"],
        ]
    )
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
    similarity_score = (
        (stats_distance * 0.2)
        + (hist_similarity * 0.25)
        + (texture_distance * 0.2)
        + (hu_distance * 0.2)
        + (lbp_distance * 0.1)
        + (bbox_diff * 0.05)
    )

    return similarity_score


def chi_square_distance(hist1, hist2):
    """Calcula la distancia Chi-cuadrado entre dos histogramas."""
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-7))


def calculate_dynamic_thresholds(database):
    """Calcula dos umbrales basados en los percentiles 80 y 95 de las similitudes previas."""
    similarity_scores = []
    values = list(database.values())

    for i in tqdm(range(len(values))):
        for j in range(i + 1, len(values)):
            similarity_scores.append(compare_cases(values[i], values[j]))

    return np.percentile(similarity_scores, [80, 95])
