import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

def create_image_data_generator():
    """
    Crée un générateur de données augmentées pour l'entraînement
    """
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Pas de flip horizontal pour les panneaux de signalisation
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )

def visualize_augmentations(image, datagen, num_samples=5):
    """
    Visualise les augmentations de données sur une image
    """
    # Expand dimensions for batch processing
    image_expanded = np.expand_dims(image, 0)
    
    # Initialize the iterator
    aug_iter = datagen.flow(image_expanded, batch_size=1)
    
    # Plot original and augmented images
    plt.figure(figsize=(12, 3))
    plt.subplot(1, num_samples+1, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    for i in range(num_samples):
        aug_image = next(aug_iter)[0].astype('float32')
        plt.subplot(1, num_samples+1, i+2)
        plt.imshow(aug_image)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def apply_augmentation(X_train, y_train, batch_size=32):
    """
    Applique l'augmentation de données aux données d'entraînement
    """
    datagen = create_image_data_generator()
    datagen.fit(X_train)
    
    # Retourne un générateur pour être utilisé dans model.fit
    return datagen.flow(X_train, y_train, batch_size=batch_size)