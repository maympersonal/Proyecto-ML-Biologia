# Flujo del pipeline de trabajo con CBRT

```mermaid
graph TD
  subgraph S1["**1. Segmentación de Imágenes**"]
    style S1 fill:#B8E3FF,stroke:#005A9C,stroke-width:2px
    direction TB
    A[Conversión a grises] --> B[Filtro Gaussiano]
    B --> C[Umbralización de Otsu]
    C --> D[Detección de contornos]
    D --> E[Bounding boxes]
  end

  S1 --> S2
  subgraph S2["**2. Extracción de Características**"]
    style S2 fill:#C3F8C3,stroke:#006400,stroke-width:2px
    F[Descriptores de forma/textura]
  end

  S2 --> S3
  subgraph S3["**3. Predicción y Clasificación**"]
    style S3 fill:#FFD8B1,stroke:#FF6600,stroke-width:2px
    direction TB
    G[Comparación con casos históricos] --> H[Selección de k-nn]
    H --> I[Clasificación por zonas de confianza]
  end

  S3 --> S4
  subgraph S4["**4. Gestión de Resultados**"]
    style S4 fill:#E6D3FF,stroke:#4B0082,stroke-width:2px
    direction TB
    J{{Decisión}} -->|Alta similitud| K[Actualizar BD]
    J -->|Incertidumbre| L[Búsqueda recursiva]
    J -->|Baja similitud| M[Intervención humana]
  end

  S4 --> S5[(**Base de Datos**)]
  S5 --> S3
  style S5 fill:#FFF3B0,stroke:#CC9900,stroke-width:2px
```

# Pipeline Vertical como solución al problema

```mermaid
graph TB
  A[Segmentación] --> B[Características]
  B --> C[Clasificación]
  C --> D[Gestión]
```

# Algoritmos alternativos a cada bloque

### **1. Segmentación de Imágenes**  
**Algoritmos actuales**:  
- Umbralización de Otsu, Detección de contornos (OpenCV)

**Alternativas/Complementos**:  

| Algoritmo | Uso | Ventajas | Desventajas |
|-----------|-----|----------|-------------|
| **CNN (U-Net/Mask R-CNN)** | Segmentación semántica/instancia | Alta precisión con datos etiquetados | Requiere gran cantidad de datos |
| **K-Means** | Clustering de píxeles | Simple para pre-segmentación | Sensible a inicialización |
| **Watershed** | Segmentación basada en marcadores | Bueno para objetos superpuestos | Requiere parámetros ajustables |
| **Superpixels (SLIC)** | División en regiones homogéneas | Reduce complejidad computacional | Menos preciso en bordes irregulares |
| **GrabCut** | Segmentación interactiva | Combina modelos gráficos y user-input | Requiere intervención humana |

---

### **2. Extracción de Características**  
**Algoritmos actuales**:  
- GLCM, LBP, Momentos de Hu, Histogramas

**Alternativas/Complementos**:  

| Algoritmo | Uso | Ventajas | Desventajas |
|-----------|-----|----------|-------------|
| **CNN (ResNet/VGG)** | Extracción automática de _features_ | Captura patrones jerárquicos | Requiere fine-tuning |
| **Autoencoder** | Reducción dimensionalidad | Aprendizaje no supervisado | Pérdida de interpretabilidad |
| **SIFT/SURF** | Descriptores locales invariantes | Robustez a transformaciones | Computacionalmente costoso |
| **HOG** | Detección de bordes orientados | Efectivo para formas | Sensible a iluminación |
| **Transformers (ViT)** | Captura contexto global | Bueno para relaciones espaciales | Alto costo computacional |

---

### **3. Similitud y Clasificación**  
**Algoritmos actuales**:  
- k-NN con métricas híbridas

**Alternativas/Complementos**:  

| Algoritmo | Uso | Ventajas | Desventajas |
|-----------|-----|----------|-------------|
| **SVM** | Clasificación lineal/no-lineal | Efectivo en espacios de alta dimensión | Sensible a parámetros de kernel |
| **Random Forest** | Clasificación ensemble | Reduce sobreajuste | Menos interpretable |
| **XGBoost/LightGBM** | Clasificación con boosting | Alta precisión en tabular data | Requiere ajuste de hiperparámetros |
| **Siamese Networks** | Aprendizaje métrico | Robustez a variaciones intra-clase | Necesita pares de entrenamiento |
| **DBSCAN** | Clustering para agrupamiento | Detecta outliers automáticamente | Sensible a parámetros ε y min_samples |

---

### **4. Gestión de Conocimiento**  
**Algoritmos actuales**:  
- CBR (Case-Based Reasoning)

**Alternativas/Complementos**:  

| Algoritmo | Uso | Ventajas | Desventajas |
|-----------|-----|----------|-------------|
| **Autoencoder + K-Means** | Clustering de casos | Reduce redundancia en la BD | Pérdida de detalles finos |
| **Graph Neural Networks** | Representación relacional | Modela conexiones entre casos | Complejidad de implementación |
| **Active Learning + SVM** | Muestreo interactivo | Optimiza anotación humana | Depende del oráculo humano |
| **Reinforcement Learning** | Actualización dinámica de BD | Aprendizaje adaptativo continuo | Costoso en recursos |
| **Ontologías** | Organización semántica | Facilita razonamiento lógico | Requiere diseño experto |


# Combinaciones Propuestas  
Cada combinación representa una configuración específica de algoritmos en los 4 bloques del pipeline:

| #   | Segmentación     | Extracción de Características | Clasificación/Similitud | Gestión de Conocimiento      |
| --- | ---------------- | ----------------------------- | ----------------------- | ---------------------------- |
| 1   | Otsu + Contornos | GLCM + Hu Moments             | k-NN ponderado          | CBR + Active Learning        |
| 2   | U-Net (CNN)      | Autoencoder + Estadísticas    | SVM (RBF Kernel)        | CBR + K-Means clustering     |
| 4   | Mask R-CNN       | SIFT + LBP                    | Random Forest           | Ontologías + Active Learning |