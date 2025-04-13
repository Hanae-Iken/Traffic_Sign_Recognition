import cv2
import numpy as np
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_image(image_path):
    """Charge une image à partir du chemin spécifié"""
    return cv2.imread(image_path)

def resize_image(image, target_size=(224, 224)):
    """Redimensionne l'image à la taille spécifiée"""
    return cv2.resize(image, target_size)

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Prétraite une image pour ResNet50:
    1. Charge l'image
    2. Convertit en RGB
    3. Redimensionne
    4. Applique le prétraitement spécifique à ResNet50
    """
    image = load_image(image_path)
    if image is None:
        print(f"Impossible de charger l'image: {image_path}")
        return None
    
    # Conversion de BGR à RGB (OpenCV charge en BGR par défaut)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionnement
    image = resize_image(image, target_size)
    
    # Prétraitement spécifique à ResNet50 (normalisation, etc.)
    image = preprocess_input(image)
    
    return image

def preprocess_for_detection(image, target_size=(224, 224)):
    """
    Prétraitement spécifique pour la détection en temps réel
    """
    # Conversion BGR à RGB si l'image est en BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionnement
    image = resize_image(image, target_size)
    
    # Prétraitement spécifique à ResNet50
    image = preprocess_input(image)
    
    return image