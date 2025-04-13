import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
class Config:
    TRAIN_PATH = "data/Train"
    TEST_CSV = "data/Test.csv"
    META_CSV = "data/Meta.csv"
    LABELS_FILE = "data/classes.csv"
    IMG_DIMS = (32, 32, 3)
    BATCH_SIZE = 64
    EPOCHS = 30
    MODEL_SAVE_PATH = "models/traffic_sign_model.keras"

def load_labels():
    """Charge les labels depuis le fichier CSV"""
    labels_df = pd.read_csv(Config.LABELS_FILE)
    return dict(zip(labels_df['ClassId'], labels_df['Name']))

def load_from_csv(csv_path, img_folder):
    """Charge les images depuis un fichier CSV"""
    df = pd.read_csv(csv_path)
    images, class_ids = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(img_folder, os.path.basename(row['Path']))
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, Config.IMG_DIMS[:2])
                images.append(img)
                class_ids.append(row['ClassId'])
    return np.array(images), np.array(class_ids)

def load_train_data(labels_dict):
    """Charge les données d'entraînement"""
    X_train, y_train = [], []
    for class_id in labels_dict.keys():
        class_path = os.path.join(Config.TRAIN_PATH, str(class_id))
        if os.path.exists(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, Config.IMG_DIMS[:2])
                    X_train.append(img)
                    y_train.append(class_id)
    return np.array(X_train), np.array(y_train)

def preprocess(img):
    """Prétraitement des images"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img / 255.0

def build_model(num_classes):
    """Construit le modèle CNN"""
    model = Sequential([
        Input(shape=(32, 32, 1)),
        Conv2D(64, (5, 5), activation='relu'),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.7),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.7),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Chargement des données
    labels_dict = load_labels()
    X_train, y_train = load_train_data(labels_dict)
    X_test, y_test = load_from_csv(Config.TEST_CSV, "data/Test")
    X_val, y_val = load_from_csv(Config.META_CSV, "data/Meta")

    # Prétraitement
    X_train = np.array([preprocess(img) for img in X_train]).reshape(-1, 32, 32, 1)
    X_test = np.array([preprocess(img) for img in X_test]).reshape(-1, 32, 32, 1)
    X_val = np.array([preprocess(img) for img in X_val]).reshape(-1, 32, 32, 1)

    # Conversion des labels
    y_train = to_categorical(y_train, len(labels_dict))
    y_test = to_categorical(y_test, len(labels_dict))
    y_val = to_categorical(y_val, len(labels_dict))

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.25,
        shear_range=0.15,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # Construction et entraînement du modèle
    model = build_model(len(labels_dict))
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=Config.BATCH_SIZE),
        epochs=Config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Sauvegarde
    model.save(Config.MODEL_SAVE_PATH)
    with open("models/labels.pkl", "wb") as f:
        pickle.dump(labels_dict, f)

    # Visualisation
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()