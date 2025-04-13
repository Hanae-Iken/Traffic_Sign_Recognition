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

# Parameters
path_train = "data/Train"  # Dossier contenant les images d'entraînement
path_test = "data/Test"    # Dossier contenant les images de test
path_meta = "data/Meta"    # Dossier contenant les images de validation
labels_file = "data/classes.csv"  # Fichier CSV avec les noms des classes

batch_size_val = 50
steps_per_epoch_val = 2000
epochs_val = 10
image_dimensions = (32, 32, 3)
test_ratio = 0.2
validation_ratio = 0.2

# Importation des images et des labels
def load_data(path, labels):
    images = []
    class_ids = []
    for class_id, class_name in labels.items():
        class_path = os.path.join(path, str(class_id))
        if os.path.exists(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (image_dimensions[0], image_dimensions[1]))
                    images.append(img)
                    class_ids.append(class_id)
    return np.array(images), np.array(class_ids)

# Charger les labels depuis le CSV
labels_df = pd.read_csv(labels_file)
labels_dict = dict(zip(labels_df['ClassId'], labels_df['Name']))

# Charger les données d'entraînement, de test et de validation
X_train, y_train = load_data(path_train, labels_dict)
X_test, y_test = load_data(path_test, labels_dict)
X_validation, y_validation = load_data(path_meta, labels_dict)

# Vérification des dimensions des données
print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)
print("Validation shape:", X_validation.shape, y_validation.shape)

# Prétraitement des images
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

X_train = np.array([preprocess(img) for img in X_train])
X_test = np.array([preprocess(img) for img in X_test])
X_validation = np.array([preprocess(img) for img in X_validation])

# Redimensionnement pour CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# Augmentation des données
data_gen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
data_gen.fit(X_train)

# Conversion des labels en catégoriels
y_train = to_categorical(y_train, len(labels_dict))
y_test = to_categorical(y_test, len(labels_dict))
y_validation = to_categorical(y_validation, len(labels_dict))

# Définition du modèle CNN
def create_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(image_dimensions[0], image_dimensions[1], 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels_dict), activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Entraînement du modèle
model = create_model()
print(model.summary())

history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=steps_per_epoch_val,
    epochs=epochs_val,
    validation_data=(X_validation, y_validation),
    shuffle=True
)

# Sauvegarde du modèle
model.save("models/traffic_sign_model.h5")
with open("models/traffic_sign_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Affichage des courbes d'apprentissage
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

# Évaluation sur le jeu de test
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", score[1])