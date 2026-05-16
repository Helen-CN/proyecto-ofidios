from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from config import *

# Transformaciones para el set de entrenamiento (con aumento de datos)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformaciones para validación y prueba (sin aumento de datos)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloaders():
    # Cargar datasets originales
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=test_transforms)
    
    # ─── CÁLCULO DE PESOS PARA MITIGAR EL DESBALANCE ───
    # Extraer las etiquetas de todo el set de entrenamiento
    target_list = torch.tensor(train_dataset.targets)
    
    # Contar cuántas imágenes hay por cada clase (bincount)
    class_count = torch.bincount(target_list)
    
    # Calcular el peso inverso (a menor cantidad de imágenes, mayor peso asignado)
    class_weights = 1.0 / class_count.float()
    
    # Asignar a cada muestra individual el peso correspondiente a su clase
    class_weights_all = class_weights[target_list]
    
    # Configurar el Sampler Equilibrado
    sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True  # Permite sobremuestrear la clase minoritaria en los batches
    )
    
    # Crear los DataLoaders
    # CRÍTICO: 'shuffle' debe ser False (o no declararse) si se utiliza un sampler en el train_loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, train_dataset.classes