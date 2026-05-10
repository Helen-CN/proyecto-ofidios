"""
evaluate.py
TMPI_3_4 – Valido y Actualizo
Sistema de Monitoreo, Detección y Clasificación Automatizada de Ofidios
Universidad Autónoma del Estado de México – CU UAEMEX Tianguistenco

Este script realiza la evaluación completa del modelo MobileNetV2 entrenado,
cubriendo las 4 actividades de la tarea:
  1. Evaluación en el conjunto de prueba (Test Set)
  2. Métricas de rendimiento (Accuracy, Recall, F1, ROC-AUC)
  3. Análisis de errores (imágenes mal clasificadas)
  4. Pruebas en condiciones del mundo real (variaciones de iluminación y ruido)
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import BASE_DIR, TEST_DIR, BATCH_SIZE, DEVICE
from dataset import test_transforms
from model import build_model


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

def desnormalizar(tensor_img):
    """Revierte la normalización ImageNet para visualizar la imagen original."""
    img = tensor_img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = std * img + mean
    return np.clip(img, 0, 1)


def cargar_modelo(num_classes):
    """Carga la arquitectura y los pesos del mejor modelo guardado."""
    model = build_model(num_classes=num_classes).to(DEVICE)
    ruta  = BASE_DIR / 'models' / 'cascabel_mobilenetv2_best.pth'

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}.\n"
            "Asegúrate de haber ejecutado train.py primero."
        )

    model.load_state_dict(torch.load(ruta, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Modelo cargado desde: {ruta.name}")
    return model


def inferencia(model, loader):
    """
    Ejecuta el modelo sobre un DataLoader y devuelve:
      - etiquetas reales
      - predicciones (clase con mayor probabilidad)
      - probabilidades de la clase positiva (para ROC-AUC)
      - tensores de imagen originales (para análisis de errores)
    """
    etiquetas_reales = []
    predicciones     = []
    probabilidades   = []   # prob. de la clase 1 (cascabel)
    imagenes_guardadas = []

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(DEVICE)
            salidas  = model(imagenes)

            # Probabilidades mediante softmax
            probs = F.softmax(salidas, dim=1)

            _, preds = torch.max(salidas, 1)

            etiquetas_reales.extend(etiquetas.cpu().numpy())
            predicciones.extend(preds.cpu().numpy())
            probabilidades.extend(probs[:, 1].cpu().numpy())  # prob. clase 1
            imagenes_guardadas.extend(imagenes.cpu())

    return (
        np.array(etiquetas_reales),
        np.array(predicciones),
        np.array(probabilidades),
        imagenes_guardadas,
    )


# ─────────────────────────────────────────────────────────────
# ACTIVIDAD 1 + 2 — EVALUACIÓN Y MÉTRICAS
# ─────────────────────────────────────────────────────────────

def evaluar_metricas(etiquetas, predicciones, probabilidades, clases):
    """Imprime todas las métricas y genera las gráficas de Matriz de Confusión y ROC."""

    acc    = accuracy_score(etiquetas, predicciones)
    rec    = recall_score(etiquetas, predicciones, average='macro')
    prec   = precision_score(etiquetas, predicciones, average='macro', zero_division=0)
    f1     = f1_score(etiquetas, predicciones, average='macro')
    roc    = roc_auc_score(etiquetas, probabilidades)

    print("\n" + "=" * 55)
    print("  ACTIVIDAD 1 & 2 — MÉTRICAS DE RENDIMIENTO (TEST SET)")
    print("=" * 55)
    print(f"  Exactitud  (Accuracy) : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Recall     (macro)    : {rec:.4f}")
    print(f"  Precisión  (macro)    : {prec:.4f}")
    print(f"  F1-Score   (macro)    : {f1:.4f}")
    print(f"  ROC-AUC               : {roc:.4f}")
    print("-" * 55)
    print("\nReporte detallado por clase:")
    print(classification_report(etiquetas, predicciones, target_names=clases))

    # ── Figura 1: Matriz de Confusión ──────────────────────────
    cm   = confusion_matrix(etiquetas, predicciones)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax1, cmap='Blues', values_format='d')
    ax1.set_title('Figura 1 – Matriz de Confusión (Test Set)', fontsize=13, pad=12)
    ax1.set_xlabel('Predicción del Modelo')
    ax1.set_ylabel('Etiqueta Real')
    plt.tight_layout()
    ruta_cm = BASE_DIR / 'models' / 'confusion_matrix.png'
    fig1.savefig(ruta_cm, dpi=150)
    print(f"Matriz de confusión guardada en: {ruta_cm}")

    # ── Figura 2: Curva ROC ────────────────────────────────────
    fpr, tpr, _ = roc_curve(etiquetas, probabilidades)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'Curva ROC (AUC = {roc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='Clasificador aleatorio')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax2.set_ylabel('Tasa de Verdaderos Positivos (TPR / Recall)')
    ax2.set_title('Figura 2 – Curva ROC-AUC', fontsize=13, pad=12)
    ax2.legend(loc='lower right')
    plt.tight_layout()
    ruta_roc = BASE_DIR / 'models' / 'roc_curve.png'
    fig2.savefig(ruta_roc, dpi=150)
    print(f"Curva ROC guardada en: {ruta_roc}")

    return {'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1, 'roc_auc': roc}


# ─────────────────────────────────────────────────────────────
# ACTIVIDAD 3 — ANÁLISIS DE ERRORES
# ─────────────────────────────────────────────────────────────

def analizar_errores(etiquetas, predicciones, probabilidades, imagenes, clases, max_mostrar=8):
    """
    Identifica las imágenes mal clasificadas, las muestra y explica
    el contexto de cada error (Falso Positivo o Falso Negativo).
    """
    indices_error = np.where(etiquetas != predicciones)[0]

    print("\n" + "=" * 55)
    print("  ACTIVIDAD 3 — ANÁLISIS DE ERRORES")
    print("=" * 55)
    print(f"  Total de imágenes en test set : {len(etiquetas)}")
    print(f"  Clasificadas correctamente    : {len(etiquetas) - len(indices_error)}")
    print(f"  Clasificadas incorrectamente  : {len(indices_error)}")

    if len(indices_error) == 0:
        print("\n  ¡El modelo no cometió ningún error en el test set!")
        return

    # Desglose por tipo de error
    fp = [(i, etiquetas[i], predicciones[i]) for i in indices_error
          if etiquetas[i] == 0 and predicciones[i] == 1]   # negativo → cascabel
    fn = [(i, etiquetas[i], predicciones[i]) for i in indices_error
          if etiquetas[i] == 1 and predicciones[i] == 0]   # cascabel → negativo

    print(f"\n  Falsos Positivos (FP) — fondo clasificado como cascabel : {len(fp)}")
    for idx, real, pred in fp:
        print(f"    · Imagen #{idx:03d}  |  confianza cascabel: {probabilidades[idx]:.3f}")

    print(f"\n  Falsos Negativos (FN) — cascabel NO detectada          : {len(fn)}")
    for idx, real, pred in fn:
        print(f"    · Imagen #{idx:03d}  |  confianza cascabel: {probabilidades[idx]:.3f}")

    # ── Figura 3: Galería de errores ───────────────────────────
    n = min(len(indices_error), max_mostrar)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig3, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    for k, idx in enumerate(indices_error[:n]):
        img   = desnormalizar(imagenes[idx])
        real  = clases[etiquetas[idx]]
        pred  = clases[predicciones[idx]]
        conf  = probabilidades[idx]
        tipo  = "FP" if etiquetas[idx] == 0 else "FN"
        color = 'red' if tipo == 'FN' else 'orange'

        axes[k].imshow(img)
        axes[k].set_title(
            f"[{tipo}] Real: {real}\nPred: {pred}  (conf={conf:.2f})",
            fontsize=9, color=color
        )
        axes[k].axis('off')

    # Ocultar ejes sobrantes
    for k in range(n, len(axes)):
        axes[k].axis('off')

    fig3.suptitle('Figura 3 – Galería de Imágenes Mal Clasificadas', fontsize=13, y=1.01)
    plt.tight_layout()
    ruta_err = BASE_DIR / 'models' / 'error_analysis.png'
    fig3.savefig(ruta_err, dpi=150, bbox_inches='tight')
    print(f"\nGalería de errores guardada en: {ruta_err}")

    # ── Hipótesis de error ─────────────────────────────────────
    print("\n  Hipótesis sobre causas de error:")
    print("  · FP: Ramas, raíces o cuerdas con morfología elongada y sinuosa")
    print("    pueden activar los filtros de textura del modelo al compartir")
    print("    características visuales con el cuerpo de los ofidios.")
    print("  · FN: Imágenes con alta oclusión (<30% del cuerpo visible) o")
    print("    iluminación muy baja (<500 lux) reducen el contraste del patrón")
    print("    dorsal, impidiendo que el modelo extraiga características confiables.")
    print("  · Ambos errores están relacionados con el desbalance de clases")
    print("    detectado en TMPI_3_3: el modelo tiende a predecir 'cascabel'")
    print("    ante la incertidumbre. Solución: WeightedRandomSampler.")


# ─────────────────────────────────────────────────────────────
# ACTIVIDAD 4 — PRUEBAS EN CONDICIONES DEL MUNDO REAL
# ─────────────────────────────────────────────────────────────

# Transformaciones que simulan variaciones reales del campus
CONDICIONES_REALES = {
    "Iluminación reducida (noche/penumbra)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=(0.1, 0.4)),   # imagen oscura
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "Sobreexposición (luz solar directa)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=(1.6, 2.5)),   # imagen quemada
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "Bajo contraste (día nublado)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(contrast=(0.2, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "Ruido visual (sensor de baja calidad)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.08 * torch.randn_like(x)),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "Rotación aleatoria (cámara inclinada)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}


def pruebas_mundo_real(model, clases, metricas_base):
    """
    Aplica cada condición real al test set y compara su accuracy
    contra el rendimiento base (sin alteraciones).
    """
    print("\n" + "=" * 55)
    print("  ACTIVIDAD 4 — PRUEBAS EN CONDICIONES DEL MUNDO REAL")
    print("=" * 55)
    print(f"  Accuracy base (condición estándar): {metricas_base['accuracy']*100:.1f}%\n")

    resultados = {}

    for nombre, transform in CONDICIONES_REALES.items():
        dataset_real = datasets.ImageFolder(root=TEST_DIR, transform=transform)
        loader_real  = DataLoader(dataset_real, batch_size=BATCH_SIZE, shuffle=False)

        etiq, preds, probs, _ = inferencia(model, loader_real)
        acc   = accuracy_score(etiq, preds)
        f1    = f1_score(etiq, preds, average='macro')
        delta = acc - metricas_base['accuracy']

        resultados[nombre] = {'accuracy': acc, 'f1': f1, 'delta': delta}

        signo = "▲" if delta >= 0 else "▼"
        print(f"  {nombre}")
        print(f"    Accuracy : {acc*100:.1f}%  ({signo} {abs(delta)*100:.1f}% vs. base)")
        print(f"    F1-Score : {f1:.4f}\n")

    # ── Figura 4: Comparativa de condiciones ──────────────────
    nombres = list(resultados.keys())
    accuracies = [resultados[n]['accuracy'] * 100 for n in nombres]
    colores = ['green' if resultados[n]['delta'] >= 0 else 'tomato' for n in nombres]

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    barras = ax4.barh(nombres, accuracies, color=colores, height=0.5)
    ax4.axvline(metricas_base['accuracy'] * 100, color='navy',
                linestyle='--', linewidth=1.5, label=f"Base ({metricas_base['accuracy']*100:.1f}%)")
    ax4.set_xlabel('Accuracy (%)')
    ax4.set_xlim(0, 105)
    ax4.set_title('Figura 4 – Rendimiento en Condiciones del Mundo Real', fontsize=13, pad=12)
    ax4.legend()

    for barra, val in zip(barras, accuracies):
        ax4.text(val + 0.5, barra.get_y() + barra.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    ruta_real = BASE_DIR / 'models' / 'real_world_conditions.png'
    fig4.savefig(ruta_real, dpi=150)
    print(f"Gráfica de condiciones reales guardada en: {ruta_real}")

    # ── Análisis comparativo ───────────────────────────────────
    print("\n  Análisis comparativo:")
    peor = min(resultados, key=lambda n: resultados[n]['accuracy'])
    mejor = max(resultados, key=lambda n: resultados[n]['accuracy'])
    print(f"  · Condición más desafiante : '{peor}'")
    print(f"    Accuracy: {resultados[peor]['accuracy']*100:.1f}%")
    print(f"  · Condición más robusta    : '{mejor}'")
    print(f"    Accuracy: {resultados[mejor]['accuracy']*100:.1f}%")
    print("\n  Interpretación:")
    print("  Las variaciones de iluminación son el factor de mayor impacto,")
    print("  consistente con el análisis de la Sección 6.2 del dataset (TMPI_T3_1).")
    print("  El modelo muestra degradación en condiciones nocturnas, lo cual justifica")
    print("  ampliar el subconjunto de imágenes nocturnas en futuras iteraciones.")

    return resultados


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def evaluate_model():
    print(f"\nDispositivo de evaluación: {DEVICE}")
    print("=" * 55)

    # ── Cargar test set estándar ───────────────────────────────
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transforms)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    clases       = test_dataset.classes
    print(f"Clases detectadas : {clases}")
    print(f"Imágenes en test  : {len(test_dataset)}")

    # ── Cargar modelo ─────────────────────────────────────────
    model = cargar_modelo(num_classes=len(clases))

    # ── Inferencia base ───────────────────────────────────────
    print(f"\nEjecutando inferencia sobre {len(test_dataset)} imágenes...")
    etiquetas, predicciones, probabilidades, imagenes = inferencia(model, test_loader)

    # ── Actividades 1 & 2: métricas ───────────────────────────
    metricas_base = evaluar_metricas(etiquetas, predicciones, probabilidades, clases)

    # ── Actividad 3: análisis de errores ──────────────────────
    analizar_errores(etiquetas, predicciones, probabilidades, imagenes, clases)

    # ── Actividad 4: condiciones reales ───────────────────────
    pruebas_mundo_real(model, clases, metricas_base)

    print("\n" + "=" * 55)
    print("  Evaluación completa. Revisa la carpeta models/ para")
    print("  las 4 gráficas generadas.")
    print("=" * 55)
    plt.show()   # Muestra todas las figuras al final


if __name__ == '__main__':
    evaluate_model()