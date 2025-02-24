from typing import List, Tuple
import numpy as np
from sklearn.svm import SVC


def svm_manager(classifications: List[Tuple[float, str]], k: int = 20) -> str | None:
    """
    Utiliza un SVM con kernel RBF para procesar los k vecinos.

    Parámetros:
      - classifications: Lista de tuplas (distancia, label).
      - k: Número de vecinos a considerar (por defecto 20).

    Retorna:
      - El label predicho si la probabilidad máxima es mayor al 40%, o None en caso contrario.
    """
    # Ordenar por la distancia (el primer valor de la tupla)
    classifications.sort(key=lambda x: x[0])
    top_k = classifications[:k]

    # Construir el conjunto de entrenamiento:
    # Se toma la distancia como característica (1D) y el label como objetivo.
    X = np.array([[d] for d, _ in top_k], dtype=np.float32)
    y = [label for _, label in top_k]

    # Entrenar el SVM con kernel RBF y habilitar la estimación de probabilidades.
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(X, y)

    # La "consulta" se representa como el ejemplo con distancia 0 (punto de referencia).
    query_feature = np.array([[0.0]], dtype=np.float32)
    prediction = svm.predict(query_feature)[0]
    probs = svm.predict_proba(query_feature)[0]

    if max(probs) > 0.4:
        return prediction
    return None
