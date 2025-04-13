import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
import pickle

# Paramètres
path_train = "data/Train"  # Dossier avec sous-dossiers par classe
path_test_csv = "data/Test.csv"  # Fichier CSV pour les tests
path_meta_csv = "data/Meta.csv"  # Fichier CSV pour la validation
labels_file = "data/classes.csv"  # Fichier des labels
image_dimensions = (32, 32, 3)
batch_size_val = 50
epochs_val = 10

# Chargement des labels
labels_df = pd.read_csv(labels_file)
labels_dict = dict(zip(labels_df['ClassId'], labels_df['Name']))

# Fonction pour charger les images depuis un CSV
def load_from_csv(csv_path, img_folder):
    df = pd.read_csv(csv_path)
    images = []
    class_ids = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_folder, row['Path'].split('/')[-1])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_dimensions[:2]))
            images.append(img)
            class_ids.append(row['ClassId'])
    return np.array(images), np.array(class_ids)

# Chargement des données
X_train, y_train = [], []
for class_id in labels_dict.keys():
    class_path = os.path.join(path_train, str(class_id))
    if os.path.exists(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (image_dimensions[:2]))
                X_train.append(img)
                y_train.append(class_id)
X_train, y_train = np.array(X_train), np.array(y_train)

X_test, y_test = load_from_csv(path_test_csv, "data/Test")
X_validation, y_validation = load_from_csv(path_meta_csv, "data/Meta")

# Vérification des données
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")
print(f"Validation: {X_validation.shape}, {y_validation.shape}")

# Prétraitement
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img / 255.0

X_train = np.array([preprocess(img) for img in X_train]).reshape(-1, 32, 32, 1)
X_test = np.array([preprocess(img) for img in X_test]).reshape(-1, 32, 32, 1)
X_validation = np.array([preprocess(img) for img in X_validation]).reshape(-1, 32, 32, 1)

# Conversion des labels
y_train = to_categorical(y_train, len(labels_dict))
y_test = to_categorical(y_test, len(labels_dict))
y_validation = to_categorical(y_validation, len(labels_dict))

# Data Augmentation
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    rotation_range=10
)
datagen.fit(X_train)

# Définition du modèle
model = Sequential([
    Conv2D(60, (5, 5), activation='relu', input_shape=(32, 32, 1)),
    Conv2D(60, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(30, (3, 3), activation='relu'),
    Conv2D(30, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(500, activation='relu'),
    Dropout(0.5),
    Dense(len(labels_dict), activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size_val),
    epochs=epochs_val,
    validation_data=(X_validation, y_validation)
)

# Sauvegarde
model.save("models/traffic_sign_model.h5")
with open("models/labels.pkl", "wb") as f:
    pickle.dump(labels_dict, f)

# Visualisation
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
