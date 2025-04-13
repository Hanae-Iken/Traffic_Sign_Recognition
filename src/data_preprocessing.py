import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Charger le fichier CSV contenant les informations des images
def load_csv_data(csv_file):
    df = pd.read_csv(csv_file)

    # Nettoyer les chemins s’ils contiennent déjà un préfixe 'Train/'
    df['Path'] = df['Path'].str.replace('Train/', '', regex=False)

    return df

# Diviser les données en train et validation (80% pour l'entraînement, 20% pour la validation)
def split_data(df, test_size=0.2):
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, val_df

# Déplacer les images dans les sous-dossiers de validation
def move_images_to_validation(train_df, val_df, source_dir='data/Train', val_dir='data/Train/Validation'):
    os.makedirs(val_dir, exist_ok=True)

    for _, row in val_df.iterrows():
        image_path = os.path.join(source_dir, row['Path'])  # row['Path'] est maintenant nettoyé
        class_name = row['ClassId']

        if os.path.exists(image_path):
            class_dir = os.path.join(val_dir, str(class_name))
            os.makedirs(class_dir, exist_ok=True)

            dest_path = os.path.join(class_dir, os.path.basename(row['Path']))
            shutil.move(image_path, dest_path)
        else:
            print(f"Image non trouvée : {image_path}")
