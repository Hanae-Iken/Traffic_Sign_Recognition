import pandas as pd
import numpy as np
import os
from src.preprocess import preprocess_image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_data_from_csv(csv_path, images_dir, target_size=(64, 64)):
    """
    Charge les données à partir d'un fichier CSV et du répertoire d'images correspondant
    """
    data = pd.read_csv(csv_path)
    
    # Initialisation des tableaux pour les images et les étiquettes
    X = []
    y = []
    
    # Parcourir chaque ligne du CSV
    for index, row in data.iterrows():
        # Construire le chemin d'accès à l'image
        if 'Path' in data.columns:
            image_path = os.path.join(images_dir, row['Path'])
        else:
            # Ajustez selon la structure de votre CSV
            image_path = os.path.join(images_dir, str(row['ClassId']), str(row['Filename']))
        
        # Prétraitement de l'image
        processed_image = preprocess_image(image_path, target_size)
        
        if processed_image is not None:
            X.append(processed_image)
            y.append(row['ClassId'])
    
    # Conversion en tableaux numpy
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def encode_labels(y_train, y_val=None, y_test=None):
    """
    Encode les étiquettes catégorielles en one-hot encoding
    """
    # Encoder les étiquettes (ClassId) en entiers consécutifs si nécessaire
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Conversion en one-hot encoding
    num_classes = len(label_encoder.classes_)
    y_train_categorical = to_categorical(y_train_encoded, num_classes)
    
    if y_val is not None and y_test is not None:
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)
        y_val_categorical = to_categorical(y_val_encoded, num_classes)
        y_test_categorical = to_categorical(y_test_encoded, num_classes)
        return y_train_categorical, y_val_categorical, y_test_categorical, label_encoder, num_classes
    
    return y_train_categorical, label_encoder, num_classes

def load_and_prepare_data(train_csv, train_dir, val_csv=None, val_dir=None, test_csv=None, test_dir=None, target_size=(64, 64)):
    """
    Charge et prépare l'ensemble du jeu de données (train, validation, test)
    """
    # Chargement des données d'entraînement
    X_train, y_train = load_data_from_csv(train_csv, train_dir, target_size)
    
    if val_csv and val_dir:
        # Chargement des données de validation
        X_val, y_val = load_data_from_csv(val_csv, val_dir, target_size)
    else:
        X_val, y_val = None, None
    
    if test_csv and test_dir:
        # Chargement des données de test
        X_test, y_test = load_data_from_csv(test_csv, test_dir, target_size)
    else:
        X_test, y_test = None, None
    
    # Encodage des étiquettes
    if X_val is not None and X_test is not None:
        y_train_cat, y_val_cat, y_test_cat, label_encoder, num_classes = encode_labels(y_train, y_val, y_test)
        return X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, label_encoder, num_classes
    
    y_train_cat, label_encoder, num_classes = encode_labels(y_train)
    return X_train, y_train_cat, label_encoder, num_classes