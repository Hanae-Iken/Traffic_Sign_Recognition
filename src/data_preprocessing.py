import pandas as pd
import numpy as np
import os
from src.preprocess import preprocess_image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data_from_csv(csv_path, images_dir, target_size=(224, 224)):
    """
    Charge les données à partir d'un fichier CSV et du répertoire d'images correspondant
    """
    data = pd.read_csv(csv_path)
    
    # Initialisation des tableaux pour les images et les étiquettes
    X = []
    y = []
    
    print(f"Chargement des données depuis {csv_path}")
    print(f"Répertoire d'images: {images_dir}")
    print(f"Nombre total d'échantillons dans le CSV: {len(data)}")
    
    # Parcourir chaque ligne du CSV
    for index, row in data.iterrows():
        # Construire le chemin d'accès à l'image
        image_path = os.path.join(images_dir, row['Path'])
        
        # Afficher quelques chemins pour débogage
        if index < 5:
            print(f"Exemple de chemin d'image: {image_path}")
        
        # Prétraitement de l'image
        processed_image = preprocess_image(image_path, target_size)
        
        if processed_image is not None:
            X.append(processed_image)
            y.append(row['ClassId'])
    
    if len(X) == 0:
        raise ValueError("Aucune image n'a pu être chargée. Vérifiez les chemins d'accès.")
    
    return np.array(X), np.array(y)

def encode_labels(y_train, y_val=None, y_test=None):
    """
    Encode les étiquettes catégorielles en one-hot encoding
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    num_classes = len(label_encoder.classes_)
    y_train_categorical = to_categorical(y_train_encoded, num_classes)
    
    if y_val is not None and y_test is not None:
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)
        y_val_categorical = to_categorical(y_val_encoded, num_classes)
        y_test_categorical = to_categorical(y_test_encoded, num_classes)
        return y_train_categorical, y_val_categorical, y_test_categorical, label_encoder, num_classes
    
    return y_train_categorical, label_encoder, num_classes

def load_and_prepare_data(train_csv, train_dir, test_csv=None, test_dir=None, val_size=0.2, target_size=(224, 224)):
    """
    Charge et prépare l'ensemble du jeu de données
    """
    # Chargement des données d'entraînement
    X_train, y_train = load_data_from_csv(train_csv, train_dir, target_size)
    
    # Séparation en train et validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )
    
    if test_csv and test_dir:
        # Chargement des données de test
        X_test, y_test = load_data_from_csv(test_csv, test_dir, target_size)
    else:
        X_test, y_test = None, None
    
    # Encodage des étiquettes
    if X_test is not None:
        y_train_cat, y_val_cat, y_test_cat, label_encoder, num_classes = encode_labels(y_train, y_val, y_test)
        return X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, label_encoder, num_classes
    
    y_train_cat, y_val_cat, label_encoder, num_classes = encode_labels(y_train, y_val)
    return X_train, y_train_cat, X_val, y_val_cat, label_encoder, num_classes