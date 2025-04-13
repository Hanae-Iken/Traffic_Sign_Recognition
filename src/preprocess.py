import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prétraiter une image (redimensionnement, normalisation, conversion)
def preprocess_image(image_path):
    # Charger l'image avec OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Impossible de charger l'image à l'emplacement: {image_path}")
    
    # Redimensionner l'image à la taille requise pour ResNet50 (224x224)
    image_resized = cv2.resize(image, (224, 224))

    # Convertir l'image de BGR à RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Normaliser l'image
    image_rgb = image_rgb / 255.0

    return image_rgb

# Charger les données avec augmentation (train) et normalisation (validation)
def load_data():
    # Créer un générateur d'images pour l'entraînement avec augmentation
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    # Créer un générateur d'images pour la validation (juste la normalisation)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Charger les données d'entraînement
    train_generator = train_datagen.flow_from_directory(
        'data/Train/',  # Dossier contenant les sous-dossiers de classes pour l'entraînement
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')  # Utilise 'categorical' pour la classification multi-classes

    # Charger les données de validation
    validation_generator = validation_datagen.flow_from_directory(
        'data/Train/Validation/',  # Le sous-dossier de validation que tu as créé
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    return train_generator, validation_generator
