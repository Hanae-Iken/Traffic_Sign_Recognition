import os
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_and_prepare_data

def build_model(input_shape, num_classes):
    """
    Construit le modèle ResNet50 avec une nouvelle tête
    """
    # Charger ResNet50 sans la couche fully-connected
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Geler les couches du modèle de base
    base_model.trainable = False
    
    # Ajouter de nouvelles couches
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Définir le nouveau modèle
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def train_new_model(input_shape, batch_size, epochs, fine_tuning_epochs, 
                   train_csv, train_dir, test_csv, test_dir, model_save_path):
    """
    Entraîne un nouveau modèle
    """
    # Charger et préparer les données
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, num_classes = load_and_prepare_data(
        train_csv, train_dir, test_csv, test_dir
    )
    
    # Construire le modèle
    model = build_model(input_shape, num_classes)
    
    # Compiler le modèle
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]
    
    # Entraînement initial
    print("Phase 1: Entraînement des nouvelles couches")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Fine-tuning
    print("Phase 2: Fine-tuning")
    # Débloquer certaines couches de ResNet50
    model = load_model(model_save_path)  # Recharger le meilleur modèle
    model.get_layer('resnet50').trainable = True
    
    for layer in model.get_layer('resnet50').layers[:100]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history_fine = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=fine_tuning_epochs,
        callbacks=callbacks
    )
    
    # Évaluation
    if X_test is not None:
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Précision sur le jeu de test: {test_acc:.4f}")
    
    # Sauvegarder le modèle final
    model.save(model_save_path)
    print(f"Modèle sauvegardé à {model_save_path}")

def main():
    print("Début du programme d'entraînement de modèle ResNet50 pour la reconnaissance de panneaux de signalisation...")
    
    # Paramètres
    input_shape = (224, 224, 3)  # Taille d'entrée pour ResNet50
    batch_size = 32
    epochs = 20
    fine_tuning_epochs = 10
    
    # Chemins des données
    train_csv = 'data/Train.csv'
    train_dir = 'data'  # Car les chemins dans le CSV incluent déjà 'Train/'
    
    test_csv = 'data/Test.csv'
    test_dir = 'data'  # Car les chemins dans le CSV incluent déjà 'Test/'
    
    model_save_path = 'models/resnet50_model.h5'
    os.makedirs('models', exist_ok=True)
    
    # Vérifier si le modèle existe déjà
    if os.path.exists(model_save_path):
        print(f"Un modèle existe déjà à {model_save_path}. Voulez-vous le remplacer? (o/n)")
        choice = input().lower()
        if choice != 'o':
            print("Utilisation du modèle existant.")
            return
    
    # Entraîner un nouveau modèle
    train_new_model(input_shape, batch_size, epochs, fine_tuning_epochs, 
                   train_csv, train_dir, test_csv, test_dir, model_save_path)
    
    print("Programme terminé avec succès!")

if __name__ == '__main__':
    main()
    