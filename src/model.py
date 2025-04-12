from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_model(num_classes):
    # Charger ResNet50 pré-entraîné sans la couche dense finale
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Ajouter une couche de pooling global
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Ajouter une couche dense pour la classification
    predictions = Dense(num_classes, activation='softmax')(x)

    # Créer le modèle
    model = Model(inputs=base_model.input, outputs=predictions)

    # Geler les couches de ResNet50 pour ne pas les réentraîner
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
