import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_image(image_path):
    # Charger l'image avec OpenCV
    image = cv2.imread(image_path)

    # Redimensionner l'image à la taille requise pour ResNet50 (224x224)
    image_resized = cv2.resize(image, (224, 224))

    # Convertir l'image de BGR à RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Normaliser l'image
    image_rgb = image_rgb / 255.0

    return image_rgb

def load_data():
    # Créer un générateur d'images pour l'entraînement
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Charger les données d'entraînement et de validation
    train_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        'data/validation/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    return train_generator, validation_generator