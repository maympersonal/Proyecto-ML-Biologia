from typing import Counter
from calculate_similarity import compare_cases
from database_utils import update_database


def find_similar_cases(new_case,image_name, database, thresholds, top_n=5):
    """Clasifica un nuevo caso según tres zonas de confianza y maneja aprendizaje activo."""
    similarities = []
    
    for espora_id, case in database.items():
        score = compare_cases(new_case, case)
        similarities.append((case["bounding_box"]["class"], score))

    # Ordenar por menor distancia (más similar)
    similarities.sort(key=lambda x: x[1])
    top_matches = similarities[:top_n]

    if not top_matches:
        return {"status": "review", "class": None, "top_matches": []}

    min_distance = top_matches[0][1]
    low_threshold, high_threshold = thresholds

    # Determinar la zona de confianza
    if min_distance <= low_threshold:
        # Zona de alta confianza: clasificación automática
        most_common = Counter([cls for cls, _ in top_matches]).most_common(1)[0][0]
        update_database(database,image_name, new_case, most_common)
        return {
            "status": "high_confidence",
            "class": most_common,
            "confidence": "auto",
            "top_matches": top_matches
        }
    elif min_distance <= high_threshold:
        # Zona de incertidumbre: modelo secundario
        secondary_class = find_similar_cases(new_case,image_name, database, thresholds, top_n*2)
        return {
            "status": "uncertainty",
            "class": secondary_class,
            "confidence": "secondary",
            "top_matches": top_matches
        }
    else:
        # Zona de revisión: aprendizaje activo
        system_recommendation = Counter([cls for cls, _ in top_matches]).most_common(1)[0][0]
        user_class = active_learning_prompt(system_recommendation, top_matches)
        if user_class is None:
            return {
                "status": "review",
                "class": None,
                "confidence": "manual",
                "top_matches": top_matches
            }
        else:
            return {
                "status": "high_confidence",
                "class": user_class,
                "confidence": "manual",
                "top_matches": top_matches
            }
def active_learning_prompt(system_recommendation, top_matches):
    """Interfaz para que los expertos validen casos difíciles."""
    print(f"Revisión requerida. Recomendación del sistema: {system_recommendation}")
    print("Casos similares de referencia:")
    for i, (cls, score) in enumerate(top_matches[:3], 1):
        print(f"{i}. Clase: {cls} - Distancia: {score:.2f}")
    
    while True:
        user_input = input("Introduzca la clase correcta (o 'skip' para omitir): ").strip()
        if user_input.lower() == 'skip':
            return None
        if user_input in {cls for cls, _ in top_matches}:
            return user_input
        print("Clase no válida. Las clases existentes son:", {cls for cls, _ in top_matches})