from src.data_preprocessing import load_csv_data, split_data, move_images_to_validation
from src.model import create_model
from src.preprocess import load_data
from tensorflow.keras.models import load_model

def main():
    # Charger les données CSV
    df = load_csv_data('data/Train.csv')

    # Diviser les données en train et validation
    train_df, val_df = split_data(df)

    # Déplacer les images dans les bons dossiers (attention : source_dir = 'data/Train')
    move_images_to_validation(train_df, val_df, source_dir='data/Train')

    # Charger les générateurs d'images pour l'entraînement et la validation
    train_generator, validation_generator = load_data()

    # Créer le modèle
    model = create_model(num_classes=43)

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
