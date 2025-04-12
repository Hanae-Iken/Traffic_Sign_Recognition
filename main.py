from src.model import create_model
from src.preprocess import load_data
from tensorflow.keras.models import load_model

def main():
    # Charger les générateurs de données
    train_generator, validation_generator = load_data()

    # Créer le modèle
    model = create_model(num_classes=43)  # Assumer qu'il y a 43 classes dans le dataset

    # Entraîner le modèle
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    # Sauvegarder le modèle entraîné
    model.save('models/resnet50_model.h5')

if __name__ == "__main__":
    main()
