import matplotlib.pyplot as plt
import numpy as np
import torchvision
from dataset import get_dataloaders

def mostrar_imagenes(tensor_img, titulo=None):
    """Des-normaliza el tensor para que matplotlib pueda dibujarlo correctamente."""
    # Convertir el tensor de PyTorch (C, H, W) a formato de imagen (H, W, C)
    img = tensor_img.numpy().transpose((1, 2, 0))
    
    # Revertir la normalización que aplicamos en dataset.py
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    
    # Recortar valores fuera del rango [0, 1] por seguridad
    img = np.clip(img, 0, 1)
    
    plt.figure(figsize=(14, 6))
    plt.imshow(img)
    if titulo is not None:
        plt.title(titulo)
    plt.axis('off')

def main():
    print("Iniciando los DataLoaders...")
    train_loader, val_loader, clases = get_dataloaders()
    
    print(f"Clases detectadas en las carpetas: {clases}")
    
    # Extraer el primer lote de entrenamiento
    imagenes, etiquetas = next(iter(train_loader))
    
    print(f"Formato del tensor de imágenes: {imagenes.shape}") # Debería ser [16, 3, 224, 224]
    print(f"Etiquetas del lote (0 y 1): {etiquetas}")
    
    # Construir una cuadrícula visual con el lote completo
    cuadricula = torchvision.utils.make_grid(imagenes)
    
    # Mapear los números (0, 1) a los nombres de las carpetas para el título
    nombres_etiquetas = [clases[etiqueta] for etiqueta in etiquetas]
    titulo = " | ".join(nombres_etiquetas[:8]) # Mostramos solo los primeros 8 nombres
    
    print("\nAbrir ventana gráfica. Cierra la ventana para terminar la ejecución.")
    mostrar_imagenes(cuadricula, titulo=f"Lote de entrenamiento: {titulo} ...")
    plt.show()

if __name__ == '__main__':
    main()