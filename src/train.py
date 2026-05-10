import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score, accuracy_score
import time
import os
from config import *
from dataset import get_dataloaders
from model import build_model

def train_model():
    print(f"Iniciando entrenamiento usando: {DEVICE}")
    
    # 1. Preparar datos y modelo
    train_loader, val_loader, classes = get_dataloaders()
    model = build_model(num_classes=len(classes)).to(DEVICE)
    
    # 2. Definir función de pérdida y optimizador
    # Usamos CrossEntropyLoss porque nuestro modelo escupe 2 valores (clase 0 y clase 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: Reduce la tasa de aprendizaje si la validación se estanca
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # 3. Variables para monitorear el progreso
    mejor_loss = float('inf')
    directorio_modelos = BASE_DIR / 'models'
    directorio_modelos.mkdir(exist_ok=True) # Crea la carpeta si no existe
    
    ruta_guardado = directorio_modelos / 'cascabel_mobilenetv2_best.pth'

    for epoch in range(NUM_EPOCHS):
        inicio_epoca = time.time()
        
        # ==========================================
        # FASE DE ENTRENAMIENTO
        # ==========================================
        model.train() # Ponemos el modelo en modo entrenamiento
        train_loss = 0.0
        
        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)
            
            # Reiniciar gradientes
            optimizer.zero_grad()
            
            # Forward pass (predicción)
            salidas = model(imagenes)
            loss = criterion(salidas, etiquetas)
            
            # Backward pass (cálculo de gradientes y actualización)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imagenes.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # ==========================================
        # FASE DE VALIDACIÓN
        # ==========================================
        model.eval() # Modo evaluación (desactiva Dropout, etc.)
        val_loss = 0.0
        todas_etiquetas = []
        todas_predicciones = []
        
        # Apagamos el cálculo de gradientes para ahorrar memoria y tiempo
        with torch.no_grad():
            for imagenes, etiquetas in val_loader:
                imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)
                
                salidas = model(imagenes)
                loss = criterion(salidas, etiquetas)
                val_loss += loss.item() * imagenes.size(0)
                
                # Obtener la clase con mayor probabilidad
                _, preds = torch.max(salidas, 1)
                
                todas_etiquetas.extend(etiquetas.cpu().numpy())
                todas_predicciones.extend(preds.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        
        # Calcular métricas usando scikit-learn
        acc = accuracy_score(todas_etiquetas, todas_predicciones)
        # Asumiendo que cascabel es la clase 0 o 1 (se calculará recall promedio o específico)
        rec = recall_score(todas_etiquetas, todas_predicciones, average='macro')
        
        # Actualizar el scheduler
        scheduler.step(val_loss)
        
        tiempo_epoca = time.time() - inicio_epoca
        
        # Imprimir progreso
        print(f"Época {epoch+1}/{NUM_EPOCHS} [{tiempo_epoca:.0f}s] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {acc:.4f} | Val Recall: {rec:.4f}")
        
        # ==========================================
        # GUARDAR EL MEJOR MODELO
        # ==========================================
        if val_loss < mejor_loss:
            mejor_loss = val_loss
            torch.save(model.state_dict(), ruta_guardado)
            print(f"  -> ¡Nuevo mejor modelo guardado en {ruta_guardado.name}!")

    print("\nEntrenamiento finalizado.")

if __name__ == '__main__':
    train_model()