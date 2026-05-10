import torch.nn as nn
from torchvision import models

def build_model(num_classes=2):
    # Cargar MobileNetV2 preentrenado
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Congelar las capas iniciales (opcional, acelera el primer entrenamiento)
    for param in model.parameters():
        param.requires_grad = False
        
    # Reemplazar la última capa clasificadora para nuestro problema (2 clases)
    # cascabel vs negativo
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5), # Ayuda a prevenir el sobreajuste
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model