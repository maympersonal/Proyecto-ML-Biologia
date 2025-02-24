from typing import List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def random_forest_manager(classifications: List[Tuple[float, str]], k: int = 20) -> str | None:
    """
    Utiliza un Random Forest para procesar los k vecinos.
    
    Parámetros:
      - classifications: Lista de tuplas (distancia, label).
      - k: Número de vecinos a considerar (por defecto 20).
    
    Retorna:
      - El label predicho si la probabilidad máxima es mayor al 40%, o None en caso contrario.
    """
    # Ordenar por la distancia
    classifications.sort(key=lambda x: x[0])
    top_k = classifications[:k]
    
    # Preparar los datos de entrenamiento: la característica es la distancia (1D)
    X = np.array([[d] for d, _ in top_k], dtype=np.float32)
    y = [label for _, label in top_k]
    
    # Entrenar el Random Forest.
    rf = RandomForestClassifier()
    rf.fit(X, y)
    
    # Predecir para el "query" con característica 0.
    query_feature = np.array([[0.0]], dtype=np.float32)
    prediction = rf.predict(query_feature)[0]
    probs = rf.predict_proba(query_feature)[0]
    
    if max(probs) > 0.4:
        return prediction
    return None
