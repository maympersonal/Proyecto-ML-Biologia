import os

import cv2
from tqdm import tqdm
import CBR_algo

image_folder = "D:\\MatCom\\4toanno\\1er_Semestre\\Machine_Learning\\Proyecto\\CBR_algorithim\\Imagenes\\dataset\\train\\images"
label_folder = "D:\\MatCom\\4toanno\\1er_Semestre\\Machine_Learning\\Proyecto\\CBR_algorithim\\Imagenes\\dataset\\train\\labels"
output_json = ".\\spore_features.json"
model = CBR_algo.CBR(5)
database = CBR_algo.load_database("D:\\MatCom\\4toanno\\1er_Semestre\\Machine_Learning\\Proyecto\\CBR_algorithim\\spore_features.json")
if not database:
 model.fit(image_folder, label_folder,output_json)
image_folder_valid = "D:\\MatCom\\4toanno\\1er_Semestre\\Machine_Learning\\Proyecto\\CBR_algorithim\\Imagenes\\dataset\\valid\\images"
image_files = {os.path.splitext(f)[0]: os.path.join(image_folder_valid, f) for f in os.listdir(image_folder_valid) if f.endswith(('.jpg', '.png', '.jpeg'))}


# # Ejecutar predicción
for image in tqdm(image_files.keys()):
    print(image_files[image])
    image1 = cv2.imread(image_files[image])
    resultados = model.predict(image1, database)
    
    #     # Mostrar resultados
    for res in resultados:
        print(f"Espora detectada en {image} - Tipo: {res[0]} ")
