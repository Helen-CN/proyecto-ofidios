import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas del dataset construidas a partir de la base (funcionará en Windows y Linux por igual)
TRAIN_DIR = str(BASE_DIR / 'data' / 'train')
VAL_DIR = str(BASE_DIR / 'data' / 'val')
TEST_DIR = str(BASE_DIR / 'data' / 'test')

# Hiperparámetros
BATCH_SIZE = 16  
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Detección de dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')