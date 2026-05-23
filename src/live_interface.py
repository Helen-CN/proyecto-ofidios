"""
live_inference.py
Sistema de Monitoreo y Detección de Ofidios

Este script despliega una interfaz gráfica para elegir entre la cámara web
o un archivo de video local para ejecutar inferencia en tiempo real.
"""

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import deque

# Importamos la configuración y el modelo de nuestro proyecto
from config import BASE_DIR, DEVICE
from model import build_model

# Transformaciones idénticas a las de validación/prueba
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def cargar_modelo():
    """Carga los pesos matemáticos del mejor modelo."""
    print(f"Cargando modelo en: {DEVICE}...")
    model = build_model(num_classes=2).to(DEVICE)
    ruta_modelo = BASE_DIR / 'models' / 'cascabel_mobilenetv2_best.pth'
    
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {ruta_modelo}")
        
    model.load_state_dict(torch.load(ruta_modelo, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def procesar_video(origen_video, modelo, umbral_alerta=0.85):
    cap = cv2.VideoCapture(origen_video)
    if not cap.isOpened():
        messagebox.showerror("Error", f"No se pudo abrir la fuente de video:\n{origen_video}")
        return

    tiempo_previo = 0
    
    # Creamos un historial para los últimos 8 fotogramas
    historial_probabilidades = deque(maxlen=8)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        tiempo_actual = time.time()
        fps = 1 / (tiempo_actual - tiempo_previo) if (tiempo_actual - tiempo_previo) > 0 else 0
        tiempo_previo = tiempo_actual

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        input_tensor = inference_transforms(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            salidas = modelo(input_tensor)
            probabilidades = F.softmax(salidas, dim=1)[0]
            prob_actual = probabilidades[0].item()
        
        # ─── AQUÍ ESTÁ EL TRUCO DE SUAVIZADO ───
        historial_probabilidades.append(prob_actual)
        prob_suave = sum(historial_probabilidades) / len(historial_probabilidades)
        # ───────────────────────────────────────

        color_caja = (0, 255, 0) 
        texto_alerta = "Despejado"
        
        # Evaluamos usando la probabilidad suavizada, no la instantánea
        if prob_suave >= umbral_alerta:
            color_caja = (0, 0, 255) 
            texto_alerta = f"¡ALERTA CASCABEL! {prob_suave*100:.1f}%"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color_caja, 10)

        # (El resto de tu lógica visual y cv2.imshow se mantiene igual)
        cv2.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), -1) 
        cv2.putText(frame, texto_alerta, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color_caja, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Monitor de Seguridad Ofidios', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# INTERFAZ GRÁFICA (GUI)
# ==========================================

class AplicacionOfidios:
    def __init__(self, root, modelo):
        self.root = root
        self.modelo = modelo
        self.root.title("Monitor de Detección de Cascabel")
        self.root.geometry("400x250")
        self.root.configure(bg="#f0f0f0")
        self.root.eval('tk::PlaceWindow . center') # Centrar ventana

        # Título en la ventana
        tk.Label(
            root, 
            text="Sistema de Detección Automática", 
            font=("Arial", 14, "bold"), 
            bg="#f0f0f0",
            pady=20
        ).pack()

        tk.Label(
            root, 
            text="Selecciona la fuente de video para analizar:", 
            font=("Arial", 10), 
            bg="#f0f0f0",
            pady=10
        ).pack()

        # Marco para los botones
        frame_botones = tk.Frame(root, bg="#f0f0f0")
        frame_botones.pack(pady=10)

   
        # Botón 2: Video
        btn_video = tk.Button(
            frame_botones, 
            text="📂 Cargar Video", 
            font=("Arial", 11), 
            width=15, 
            cursor="hand2",
            command=self.cargar_video
        )
        btn_video.grid(row=0, column=1, padx=10)

    def iniciar_camara(self):
        self.root.withdraw() # Ocultar el menú
        procesar_video(0, self.modelo)
        self.root.deiconify() # Mostrar el menú nuevamente al terminar

    def cargar_video(self):
        # Abrir explorador de archivos
        ruta_video = filedialog.askopenfilename(
            title="Seleccionar Video",
            filetypes=[("Archivos de Video", "*.mp4 *.avi *.mkv *.mov"), ("Todos los archivos", "*.*")]
        )
        
        if ruta_video: # Si el usuario no canceló la selección
            self.root.withdraw() 
            procesar_video(ruta_video, self.modelo)
            self.root.deiconify() 

if __name__ == '__main__':
    # 1. Cargar el modelo en memoria primero para que la interfaz sea rápida
    try:
        modelo_cargado = cargar_modelo()
    except Exception as e:
        print(f"Error al iniciar: {e}")
        exit()

    # 2. Inicializar la ventana de Tkinter
    ventana_principal = tk.Tk()
    app = AplicacionOfidios(ventana_principal, modelo_cargado)
    
    # 3. Mantener la ventana abierta
    ventana_principal.mainloop()