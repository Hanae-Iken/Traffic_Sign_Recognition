import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, save_model
from src.data_preprocessing import load_and_prepare_data
from src.data_augmentation import apply_augmentation
from src.model import build_resnet50_model, compile_model, train_model, evaluate_model, plot_training_history
from src.predict import TrafficSignClassifier
import cv2

def main():
    print("Début du programme d'entraînement de modèle ResNet50 pour la reconnaissance de panneaux de signalisation...")
    
    # Paramètres
    input_shape = (64, 64, 3)  # Hauteur, largeur, canaux
    batch_size = 32
    epochs = 20
    fine_tuning_epochs = 10
    train_csv = 'data/Train.csv'
    train_dir = 'data/Train'
    val_csv = 'data/Meta.csv'  # Utilisez votre fichier CSV pour la validation
    val_dir = 'data/Meta'  # Utilisez votre dossier pour la validation
    test_csv = 'data/Test.csv'
    test_dir = 'data/Test'
    model_save_path = 'models/resnet50_model.h5'
    
    # Vérifier si le modèle existe déjà
    if os.path.exists(model_save_path):
        print(f"Un modèle existe déjà à {model_save_path}. Voulez-vous le remplacer? (o/n)")
        choice = input().lower()
        if choice != 'o':
            print("Utilisation du modèle existant.")
            model = load_model(model_save_path)
            # Continuer avec l'évaluation ou la prédiction
        else:
            # Entraîner un nouveau modèle
            train_new_model(input_shape, batch_size, epochs, fine_tuning_epochs, 
                           train_csv, train_dir, val_csv, val_dir, test_csv, test_dir, model_save_path)
    else:
        # Entraîner un nouveau modèle
        train_new_model(input_shape, batch_size, epochs, fine_tuning_epochs, 
                       train_csv, train_dir, val_csv, val_dir, test_csv, test_dir, model_save_path)
    
    # Test avec la caméra (décommentez pour tester)
    # test_with_camera(model_save_path, val_csv)
    
    print("Programme terminé avec succès!")

def train_new_model(input_shape, batch_size, epochs, fine_tuning_epochs, 
                   train_csv, train_dir, val_csv, val_dir, test_csv, test_dir, model_save_path):
    """
    Entraîne un nouveau modèle ResNet50
    """
    print("Chargement et préparation des données...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, num_classes = load_and_prepare_data(
        train_csv, train_dir, val_csv, val_dir, test_csv, test_dir
    )
    
    print(f"Données chargées: {X_train.shape[0]} échantillons d'entraînement, {X_val.shape[0]} échantillons de validation, {X_test.shape[0]} échantillons de test")
    print(f"Nombre de classes: {num_classes}")
    
    # Afficher quelques exemples d'images
    display_samples(X_train, y_train, label_encoder, num_samples=5)
    
    # Construction du modèle
    print("Construction du modèle ResNet50...")
    model = build_resnet50_model(input_shape, num_classes)
    model = compile_model(model)
    
    # Résumé du modèle
    model.summary()
    
    # Entraînement du modèle
    print("Entraînement du modèle...")
    augmented_data = apply_augmentation(X_train, y_train, batch_size)
    model, history = train_model(
        model, X_train, y_train, X_val, y_val, 
        batch_size, epochs, fine_tuning_epochs, model_save_path
    )
    
    # Tracer les courbes d'apprentissage
    plot_training_history(history)
    
    # Évaluer le modèle sur les données de test
    print("Évaluation du modèle sur les données de test...")
    evaluate_model(model, X_test, y_test)
    
    # Sauvegarder le mapping des classes
    class_mapping = pd.DataFrame({
        'ClassId': label_encoder.classes_,
        'EncodedId': range(len(label_encoder.classes_))
    })
    class_mapping.to_csv('models/class_mapping.csv', index=False)
    
    print(f"Modèle entraîné et sauvegardé avec succès à {model_save_path}")
    
    return model

def display_samples(X, y, label_encoder, num_samples=5):
    """
    Affiche quelques exemples d'images du jeu de données
    """
    plt.figure(figsize=(15, 3))
    indices = np.random.choice(range(len(X)), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X[idx])
        
        # Récupérer l'étiquette originale
        true_label_idx = np.argmax(y[idx])
        original_label = label_encoder.classes_[true_label_idx]
        
        plt.title(f"Class: {original_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_with_camera(model_path, label_map_path=None):
    """
    Test du modèle avec la webcam
    """
    # Initialiser le classificateur
    classifier = TrafficSignClassifier(model_path, label_map_path)
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam.")
        return
    
    while True:
        # Lire une frame
        ret, frame = cap.read()
        
        if not ret:
            print("Impossible de lire la frame de la webcam.")
            break
        
        # Traiter la frame
        annotated_frame, detection_info = classifier.process_frame(frame)
        
        # Afficher la frame annotée
        cv2.imshow('Traffic Sign Detection', annotated_frame)
        
        # Sortir avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
