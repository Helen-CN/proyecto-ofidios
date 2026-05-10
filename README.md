# 🐍 Sistema de Monitoreo y Detección de Ofidios mediante Visión Artificial

**Universidad Autónoma del Estado de México — Centro Universitario UAEMEX Tianguistenco**  
Materia: Técnicas y Métodos de Procesamiento de Imágenes — Octavo Semestre

---

## 📌 Descripción

Sistema de clasificación binaria para detectar automáticamente serpientes de cascabel (*Crotalus* spp.) en imágenes capturadas en el campus universitario, usando una red neuronal convolucional MobileNetV2 con transfer learning.

El sistema distingue entre:
- **Clase positiva:** Serpientes de cascabel
- **Clase negativa:** Fondos, vegetación y otras serpientes no venenosas

---

## 🎯 Resultados obtenidos

| Métrica | Valor |
|---|---|
| Accuracy | 89.3% |
| Recall (macro) | 0.9059 |
| F1-Score (macro) | 0.8903 |
| ROC-AUC | 0.9911 |

---

## 🗂️ Estructura del proyecto

```
proyecto/
├── src/
│   ├── config.py        # Rutas e hiperparámetros
│   ├── dataset.py       # Carga y transformaciones del dataset
│   ├── model.py         # Arquitectura MobileNetV2
│   ├── train.py         # Entrenamiento del modelo
│   ├── evaluate.py      # Evaluación completa con métricas y gráficas
│   └── test_dataset.py  # Verificación visual del dataset
├── data/
│   ├── train/
│   │   ├── cascabel/    # 70% imágenes positivas
│   │   └── negativo/    # 70% imágenes negativas
│   ├── val/
│   │   ├── cascabel/    # 15% imágenes positivas
│   │   └── negativo/    # 15% imágenes negativas
│   └── test/
│       ├── cascabel/    # 15% imágenes positivas
│       └── negativo/    # 15% imágenes negativas
├── models/              # Pesos del modelo entrenado (.pth) — no incluido en repo
├── docs/                # Documentos de entrega por tarea
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/Helen-CN/proyecto-ofidios.git
cd proyecto-ofidios
```

### 2. Crear entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 🚀 Uso

### Verificar que el dataset carga correctamente
```bash
python src/test_dataset.py
```

### Entrenar el modelo
```bash
python src/train.py
```

### Evaluar el modelo
```bash
python src/evaluate.py
```

La evaluación genera automáticamente 4 gráficas en la carpeta `models/`:
- `confusion_matrix.png` — Matriz de confusión
- `roc_curve.png` — Curva ROC-AUC
- `error_analysis.png` — Galería de imágenes mal clasificadas
- `real_world_conditions.png` — Rendimiento en condiciones reales

---

## 🧠 Arquitectura del modelo

- **Base:** MobileNetV2 preentrenado en ImageNet
- **Modificación:** Capa clasificadora personalizada con Dropout(0.5) + Linear(1280, 2)
- **Entrenamiento:** Solo la capa clasificadora (backbone congelado)
- **Optimizador:** Adam (lr=0.001)
- **Scheduler:** ReduceLROnPlateau (patience=2, factor=0.5)
- **Épocas:** 10

---

## 📊 Dataset

El dataset fue construido siguiendo el protocolo definido en TMPI_T3_1:

- **Imágenes positivas:** ~150 imágenes de serpientes de cascabel en distintas condiciones
- **Imágenes negativas:** ~203 imágenes de fondos, raíces, ramas y otras serpientes
- **División:** 70% entrenamiento / 15% validación / 15% prueba
- **Fuentes:** iNaturalist, Kaggle Snake Dataset, capturas propias en el campus

> ⚠️ Las imágenes no están incluidas en este repositorio por restricciones de tamaño y derechos.

---

## 📁 Entregas académicas

| Tarea | Descripción | Estado |
|---|---|---|
| TMPI_T3_1 | Dataset, entorno controlado y protocolo de captura | ✅ |
| TMPI_3_3 | Selección de modelo, entrenamiento y diagnóstico | ✅ |
| TMPI_3_4 | Evaluación, métricas, análisis de errores y pruebas reales | ✅ |

---

## 👥 Equipo

- Andrea Samanta Nava Baltasares
- Helen Chávez Neri
- Diego Alarcón De Jesús

**Profesora:** Rocío Elizabeth Pulido Alba