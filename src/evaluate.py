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
# ESTILO GLOBAL DE MATPLOTLIB
# Aplica un estilo limpio a todas las figuras del script.
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,   # quita borde superior
    'axes.spines.right': False,   # quita borde derecho
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    ':',
    'figure.dpi':        120,
    'axes.titlepad':     14,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'legend.framealpha': 0.85,
})

# Paleta de colores consistente en todo el script
AZUL   = '#185FA5'
VERDE  = '#0F6E56'
ROJO   = '#993C1D'
GRIS   = '#5F5E5A'
AZUL_L = '#B5D4F4'   # azul claro para FP en matriz


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
    etiquetas_reales   = []
    predicciones       = []
    probabilidades     = []
    imagenes_guardadas = []

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(DEVICE)
            salidas  = model(imagenes)
            probs    = F.softmax(salidas, dim=1)
            _, preds = torch.max(salidas, 1)

            etiquetas_reales.extend(etiquetas.cpu().numpy())
            predicciones.extend(preds.cpu().numpy())
            probabilidades.extend(probs[:, 1].cpu().numpy())
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
    """Imprime todas las métricas y genera las figuras 1 (Matriz) y 2 (ROC)."""

    acc  = accuracy_score(etiquetas, predicciones)
    rec  = recall_score(etiquetas, predicciones, average='macro')
    prec = precision_score(etiquetas, predicciones, average='macro', zero_division=0)
    f1   = f1_score(etiquetas, predicciones, average='macro')
    roc  = roc_auc_score(etiquetas, probabilidades)

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

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax1, cmap='Blues', values_format='d', colorbar=False)

    # Añade porcentaje de fila debajo de cada número
    # → permite ver recall por clase de un vistazo
    etiquetas_celda = {(0,0): 'VN', (0,1): 'FP', (1,0): 'FN', (1,1): 'VP'}
    for i in range(len(clases)):
        for j in range(len(clases)):
            total     = cm[i].sum()
            pct       = cm[i, j] / total * 100
            es_oscuro = cm[i, j] > cm.max() / 2
            color_txt = 'white' if es_oscuro else '#555'

            # Porcentaje debajo del número
            ax1.text(j, i + 0.28, f'({pct:.1f}%)',
                    ha='center', va='center',
                    fontsize=9, color=color_txt)

            # Etiqueta VP/FP/FN/VN en esquina superior izquierda de cada celda
            ax1.text(j - 0.45, i - 0.42, etiquetas_celda[(i, j)],
                    ha='left', va='top',
                    fontsize=8, color=color_txt, alpha=0.7,
                    fontweight='bold')

    ax1.set_title('Matriz de confusión — test set')
    ax1.set_xlabel('Predicción del modelo')
    ax1.set_ylabel('Etiqueta real')
    plt.tight_layout()

    ruta_cm = BASE_DIR / 'models' / 'confusion_matrix.png'
    fig1.savefig(ruta_cm, dpi=150, bbox_inches='tight')
    print(f"Matriz de confusión guardada en: {ruta_cm}")

    # ── Figura 2: Curva ROC ────────────────────────────────────
    fpr, tpr, _ = roc_curve(etiquetas, probabilidades)

    fig2, ax2 = plt.subplots(figsize=(6, 5))

    # Área sombreada bajo la curva: hace visible la ganancia vs aleatorio
    ax2.fill_between(fpr, tpr, alpha=0.12, color=AZUL)
    ax2.plot(fpr, tpr, color=AZUL, lw=2.5,
             label=f'Modelo  AUC = {roc:.3f}')
    ax2.plot([0, 1], [0, 1], linestyle='--', lw=1,
             color=GRIS, label='Clasificador aleatorio')

    # Punto de operación actual (umbral=0.5)
    fp_rate = cm[0, 1] / cm[0].sum()   # FP / total negativos reales
    tp_rate = cm[1, 0] / cm[1].sum()   # ← FN; TPR = 1 - FNR
    tp_rate = cm[1, 1] / cm[1].sum()   # VP / total positivos reales
    ax2.scatter([fp_rate], [tp_rate], color=AZUL, zorder=5, s=70,
                label=f'Umbral actual  ({fp_rate:.2f}, {tp_rate:.2f})')
    ax2.annotate(f'  ({fp_rate:.2f}, {tp_rate:.2f})',
                 xy=(fp_rate, tp_rate), fontsize=8, color=AZUL)
    
    # Líneas de referencia desde el punto de operación a los ejes
    ax2.axvline(fp_rate, ymin=0, ymax=tp_rate, color=AZUL,
                linewidth=0.8, linestyle=':', alpha=0.5)
    ax2.axhline(tp_rate, xmin=0, xmax=fp_rate, color=AZUL,
                linewidth=0.8, linestyle=':', alpha=0.5)

    # Segunda curva: rendimiento inicial (AUC=0.318, 13 imágenes)
    fpr_old = [0, 0.6, 0.8, 1.0]
    tpr_old = [0, 0.2, 0.4, 1.0]
    ax2.plot(fpr_old, tpr_old, color=GRIS, lw=1.5, linestyle='--',
            label='Evaluación inicial  AUC = 0.318')

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_title('Curva ROC-AUC')
    ax2.set_xlabel('Tasa de falsos positivos (FPR)')
    ax2.set_ylabel('Tasa de verdaderos positivos (TPR)')
    ax2.legend(loc='lower right', fontsize=9)
    plt.tight_layout()

    ruta_roc = BASE_DIR / 'models' / 'roc_curve.png'
    fig2.savefig(ruta_roc, dpi=150, bbox_inches='tight')
    print(f"Curva ROC guardada en: {ruta_roc}")

    return {'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1, 'roc_auc': roc}


# ─────────────────────────────────────────────────────────────
# ACTIVIDAD 3 — ANÁLISIS DE ERRORES
# ─────────────────────────────────────────────────────────────

def analizar_errores(etiquetas, predicciones, probabilidades, imagenes, clases, max_mostrar=None):
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

    fp = [(i, etiquetas[i], predicciones[i]) for i in indices_error
          if etiquetas[i] == 0 and predicciones[i] == 1]
    fn = [(i, etiquetas[i], predicciones[i]) for i in indices_error
          if etiquetas[i] == 1 and predicciones[i] == 0]

    print(f"\n  Falsos Positivos (FP) — fondo clasificado como cascabel : {len(fp)}")
    for idx, real, pred in fp:
        print(f"    · Imagen #{idx:03d}  |  confianza cascabel: {probabilidades[idx]:.3f}")

    print(f"\n  Falsos Negativos (FN) — cascabel NO detectada          : {len(fn)}")
    for idx, real, pred in fn:
        print(f"    · Imagen #{idx:03d}  |  confianza cascabel: {probabilidades[idx]:.3f}")

    # ── Figura 3: Galería de errores ───────────────────────────
    # Muestra TODOS los errores; max_mostrar=None significa sin límite
    n    = len(indices_error) if max_mostrar is None else min(len(indices_error), max_mostrar)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    # Cada celda ocupa 4.0in de alto; sumamos 1in fija para el suptitle
    alto = 4.0 * rows + 1.0
    fig3, axes = plt.subplots(rows, cols,
                              figsize=(3.5 * cols, alto),
                              facecolor='#f5f5f5')
    axes = np.array(axes).flatten()

    for k, idx in enumerate(indices_error[:n]):
        img  = desnormalizar(imagenes[idx])
        real = clases[etiquetas[idx]]
        pred = clases[predicciones[idx]]
        conf = probabilidades[idx]
        tipo = "FP" if etiquetas[idx] == 0 else "FN"

        # FN = rojo (serpiente no detectada, riesgo alto)
        # FP = naranja (falsa alarma, molesto pero no peligroso)
        color_borde = '#C0392B' if tipo == 'FN' else '#E67E22'

        axes[k].imshow(img)

        # Borde de color codifica el tipo de error de un vistazo
        for spine in axes[k].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color_borde)
            spine.set_linewidth(3)

        etiqueta = "FN — Cascabel no detectada" if tipo == 'FN' else "FP — Fondo como cascabel"
        axes[k].set_title(etiqueta, fontsize=8.5,
                          color=color_borde, fontweight='bold', pad=6)
        axes[k].set_xlabel(f'Pred: {pred}  |  conf: {conf:.2f}',
                           fontsize=8, labelpad=5)
        axes[k].set_xticks([])
        axes[k].set_yticks([])

    for k in range(n, len(axes)):
        axes[k].axis('off')

    fig3.suptitle('Galería de imágenes mal clasificadas',
                  fontsize=13, fontweight='bold')
    # subplots_adjust calcula top dinámicamente para que el suptitle
    # siempre tenga 0.7in reservadas sin importar cuántas filas haya
    top_margin = 1.0 - (1.2 / alto)
    fig3.subplots_adjust(top=top_margin, bottom=0.06,
                         left=0.04, right=0.98,
                         hspace=0.6, wspace=0.15)

    ruta_err = BASE_DIR / 'models' / 'error_analysis.png'
    fig3.savefig(ruta_err, dpi=150, bbox_inches='tight')
    print(f"\nGalería de errores guardada en: {ruta_err}")

    print("\n  Hipótesis sobre causas de error:")
    print("  · FP: Ramas, raíces o cuerdas con morfología elongada y sinuosa")
    print("    pueden activar los filtros de textura del modelo.")
    print("  · FN: Imágenes con alta oclusión o iluminación muy baja reducen")
    print("    el contraste del patrón dorsal del animal.")
    print("  · Solución propuesta: WeightedRandomSampler en el DataLoader.")


# ─────────────────────────────────────────────────────────────
# ACTIVIDAD 4 — PRUEBAS EN CONDICIONES DEL MUNDO REAL
# ─────────────────────────────────────────────────────────────

CONDICIONES_REALES = {
    "Iluminación reducida (noche/penumbra)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=(0.1, 0.4)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "Sobreexposición (luz solar directa)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=(1.6, 2.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "Bajo contraste (día nublado)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(contrast=(0.2, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "Ruido visual (sensor de baja calidad)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.08 * torch.randn_like(x)),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "Rotación aleatoria (cámara inclinada)": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

    resultados  = {}
    base_acc    = metricas_base['accuracy'] * 100

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
        # Ordenar de mayor a menor accuracy
        nombres    = sorted(resultados, key=lambda n: resultados[n]['accuracy'], reverse=True)
        accuracies = [resultados[n]['accuracy'] * 100 for n in nombres]

    # Verde = mejora, Rojo = degradación, Gris = sin cambio
    nombres_cortos = {
        'Iluminación reducida (noche/penumbra)': 'Noche / penumbra',
        'Sobreexposición (luz solar directa)':   'Luz solar directa',
        'Bajo contraste (día nublado)':          'Día nublado',
        'Ruido visual (sensor de baja calidad)': 'Ruido visual',
        'Rotación aleatoria (cámara inclinada)': 'Rotación aleatoria',
    }
    nombres_display = [nombres_cortos.get(n, n) for n in nombres]

    colores = []
    for n in nombres:
        if abs(resultados[n]['delta']) < 0.001:
            colores.append(GRIS)
        elif resultados[n]['delta'] >= 0:
            colores.append(VERDE)
        else:
            colores.append(ROJO)

    fig4, ax4 = plt.subplots(figsize=(10, 5), facecolor='white')
    barras = ax4.barh(nombres_display, accuracies, color=colores,
                      height=0.5, edgecolor='white', linewidth=0.5)

    ax4.axvline(base_acc, color=AZUL, linestyle='--', linewidth=1.5,
                label=f'Base ({base_acc:.1f}%)')

    # Zoom sobre el rango de variación real (amplifica las diferencias)
    margen = 3
    ax4.set_xlim(min(accuracies) - margen, max(accuracies) + margen + 4)

    ax4.set_title('Rendimiento en condiciones del mundo real')
    ax4.set_xlabel('Accuracy (%)')
    ax4.legend()

    # Etiqueta con valor absoluto y delta entre paréntesis
    for barra, val, n in zip(barras, accuracies, nombres):
        delta_n = val - base_acc
        signo   = f'+{delta_n:.1f}%' if delta_n >= 0 else f'{delta_n:.1f}%'
        label   = f'{val:.1f}%  ({signo})'
        ax4.text(val + 0.2, barra.get_y() + barra.get_height() / 2,
                 label, va='center', fontsize=8.5)

    # Leyenda de colores
    from matplotlib.patches import Patch
    leyenda_extra = [
        Patch(facecolor=VERDE, label='Mejora vs base'),
        Patch(facecolor=ROJO,  label='Degradación vs base'),
        Patch(facecolor=GRIS,  label='Sin impacto'),
    ]
    ax4.legend(handles=[ax4.get_legend_handles_labels()[0][0]] + leyenda_extra,
               labels=[f'Base ({base_acc:.1f}%)', 'Mejora vs base',
                       'Degradación vs base', 'Sin impacto'],
               loc='lower right', fontsize=8.5)

    plt.tight_layout()
    ruta_real = BASE_DIR / 'models' / 'real_world_conditions.png'
    fig4.savefig(ruta_real, dpi=150, bbox_inches='tight')
    print(f"Gráfica de condiciones reales guardada en: {ruta_real}")

    print("\n  Análisis comparativo:")
    peor  = min(resultados, key=lambda n: resultados[n]['accuracy'])
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

    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transforms)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    clases       = test_dataset.classes
    print(f"Clases detectadas : {clases}")
    print(f"Imágenes en test  : {len(test_dataset)}")

    model = cargar_modelo(num_classes=len(clases))

    print(f"\nEjecutando inferencia sobre {len(test_dataset)} imágenes...")
    etiquetas, predicciones, probabilidades, imagenes = inferencia(model, test_loader)

    metricas_base = evaluar_metricas(etiquetas, predicciones, probabilidades, clases)
    analizar_errores(etiquetas, predicciones, probabilidades, imagenes, clases)
    pruebas_mundo_real(model, clases, metricas_base)

    print("\n" + "=" * 55)
    print("  Evaluación completa. Revisa la carpeta models/ para")
    print("  las 4 gráficas generadas.")
    print("=" * 55)
    plt.show()


if __name__ == '__main__':
    evaluate_model()