import json


def load_database(json_path):
    """Carga la base de datos de esporas desde un JSON."""
    try:
        with open(json_path, "r") as file:
            return json.load(file)
    except:
        return


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


def update_database(database, image_name, new_case, user_feedback):
    """Actualiza la base de datos con nuevos casos verificados (aprendizaje activo)."""
    new_case["bounding_box"]["class"] = user_feedback
    database[image_name] = new_case
    return database
