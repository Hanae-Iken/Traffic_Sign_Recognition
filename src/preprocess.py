import cv2
import numpy as np
import os

def load_image(image_path):
    """
    Charge une image à partir du chemin spécifié
    """
    return cv2.imread(image_path)

def resize_image(image, target_size=(64, 64)):
    """
    Redimensionne l'image à la taille spécifiée
    """
    return cv2.resize(image, target_size)

def normalize_image(image):
    """
    Normalise les valeurs des pixels entre 0 et 1
    """
    return image / 255.0

def preprocess_image(image_path, target_size=(64, 64)):
    """
    Prétraite une image en la chargeant, la redimensionnant et la normalisant
    """
    image = load_image(image_path)
    if image is None:
        print(f"Impossible de charger l'image: {image_path}")
        return None
    
    # Conversion de BGR à RGB (OpenCV charge en BGR par défaut)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionnement
    image = resize_image(image, target_size)
    
    # Normalisation
    image = normalize_image(image)
    
    return image

def apply_clahe(image):
    """
    Applique CLAHE (Contrast Limited Adaptive Histogram Equalization) à l'image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def preprocess_for_detection(image, target_size=(64, 64)):
    """
    Prétraitement spécifique pour la détection en temps réel
    """
    # Conversion BGR à RGB si l'image est en BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionnement
    image = resize_image(image, target_size)
    
    # Normalisation
    image = normalize_image(image)
    
    return image