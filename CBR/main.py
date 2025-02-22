import os
import cv2
from tqdm import tqdm
from CBR_model import CBR_model
from calculate_similarity import calculate_dynamic_thresholds
from database_utils import load_database

try:
    image_folder = ".\\Imágenes\\dataset\\train\\images"
    label_folder = ".\\Imágenes\\dataset\\train\\labels"
    output_json = ".\\spore_features.json"
except:
    image_folder = "../Imágenes/dataset/train/images"
    label_folder = "../Imágenes/dataset/train/labels"
    output_json = "../spore_features.json"

model = CBR_model(5)

# Cargar base de datos
try:
    database = load_database(
        "D:\\MatCom\\4toanno\\1er_Semestre\\Machine_Learning\\Proyecto\\CBR_algorithim\\spore_features.json"
    )
except:
    database = load_database("../spore_features.json")

if not database:
    model.fit(image_folder, label_folder, output_json)

# Cargar imagen
try:
    valid_image_folder = "D:\\MatCom\\4toanno\\1er_Semestre\\Machine_Learning\\Proyecto\\CBR_algorithim\\Imagenes\\dataset\\valid\\images"
    valid_image_files = {
        os.path.splitext(f)[0]: os.path.join(valid_image_folder, f)
        for f in os.listdir(valid_image_folder)
        if f.endswith((".jpg", ".png", ".jpeg"))
    }
except:
    valid_image_folder = "../Imágenes/dataset/valid/images"
    valid_image_files = {
        os.path.splitext(f)[0]: os.path.join(valid_image_folder, f)
        for f in os.listdir(valid_image_folder)
        if f.endswith((".jpg", ".png", ".jpeg"))
    }

threshold = calculate_dynamic_thresholds(database)
# threshold = (7, 8)

# # Ejecutar predicción
for image in tqdm(valid_image_files.keys()):
    image_name = valid_image_files[image]
    image1 = cv2.imread(image_name)
    resultados = model.predict(image1, image_name, database, threshold)

    #     # Mostrar resultados
    for res in resultados:
        print(f"Espora detectada en {image} - Tipo: {res[0]} ")
