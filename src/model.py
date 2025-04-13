import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

def build_resnet50_model(input_shape, num_classes):
    """
    Construit un modèle ResNet50 personnalisé pour la classification des panneaux de signalisation
    """
    # Charger le modèle de base ResNet50 pré-entraîné sur ImageNet
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Ajouter des couches personnalisées sur le dessus du modèle de base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Construire le modèle final
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Geler les couches du modèle de base (pas d'entraînement pour ces couches)
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def unfreeze_top_layers(model, layers_to_unfreeze=30):
    """
    Dégèle les dernières couches du modèle pour le fine-tuning
    """
    # Dégeler les dernières couches pour le fine-tuning
    for layer in model.layers[-layers_to_unfreeze:]:
        layer.trainable = True
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile le modèle avec l'optimiseur, la fonction de perte et les métriques
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=20, fine_tuning_epochs=10, model_save_path='models/resnet50_model.h5'):
    """
    Entraîne le modèle en deux phases : transfert learning puis fine-tuning
    """
    # Créer le dossier de sauvegarde si nécessaire
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Callbacks pour améliorer l'entraînement
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Phase 1: Transfert learning (couches de base gelées)
    print("Phase 1: Transfert learning (couches de base gelées)")
    history_initial = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning (dégeler les dernières couches)
    print("Phase 2: Fine-tuning (couches supérieures dégelées)")
    model = unfreeze_top_layers(model)
    model = compile_model(model, learning_rate=0.0001)  # Learning rate plus faible pour le fine-tuning
    
    history_fine_tune = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=fine_tuning_epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Combiner les historiques d'entraînement
    total_history = {}
    for key in history_initial.history.keys():
        total_history[key] = history_initial.history[key] + history_fine_tune.history[key]
    
    return model, total_history

def plot_training_history(history):
    """
    Trace les courbes d'apprentissage (précision et perte)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Courbe de précision
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Courbe de perte
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur les données de test
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss, test_acc