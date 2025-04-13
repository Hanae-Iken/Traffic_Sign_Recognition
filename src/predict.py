import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.preprocess import preprocess_for_detection
import pandas as pd
import matplotlib.pyplot as plt

class TrafficSignClassifier:
    def __init__(self, model_path, label_map_path=None):
        """
        Initialise le classificateur de panneaux de signalisation
        
        Args:
            model_path: Chemin vers le modèle entraîné (.h5)
            label_map_path: Chemin vers un fichier CSV contenant le mapping des classes
        """
        self.model = load_model(model_path)
        self.target_size = (64, 64)  # Taille d'image attendue par le modèle
        
        # Chargement du mapping des classes si fourni
        self.label_map = None
        if label_map_path:
            self.load_label_map(label_map_path)
    
    def load_label_map(self, label_map_path):
        """
        Charge le mapping des ID de classe vers leurs descriptions
        """
        try:
            label_data = pd.read_csv(label_map_path)
            self.label_map = dict(zip(label_data['ClassId'], label_data['SignName']))
        except Exception as e:
            print(f"Erreur lors du chargement du mapping des classes: {e}")
            self.label_map = None
    
    def predict(self, image):
        """
        Fait une prédiction sur une image
        
        Args:
            image: Une image OpenCV ou un chemin vers une image
        
        Returns:
            class_id: ID de la classe prédite
            confidence: Confiance de la prédiction
            sign_name: Nom du panneau (si label_map est disponible)
        """
        # Vérifier si l'entrée est un chemin vers une image
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prétraitement de l'image
        processed_image = preprocess_for_detection(image, self.target_size)
        
        # Ajouter une dimension pour le batch
        input_image = np.expand_dims(processed_image, axis=0)
        
        # Prédiction
        predictions = self.model.predict(input_image)[0]
        class_id = np.argmax(predictions)
        confidence = predictions[class_id]
        
        # Obtenir le nom du panneau si le mapping est disponible
        sign_name = self.label_map.get(class_id, 'Unknown') if self.label_map else 'Unknown'
        
        return class_id, confidence, sign_name
    
    def process_frame(self, frame, detection_threshold=0.7):
        """
        Traite une frame de vidéo et détecte les panneaux de signalisation
        
        Args:
            frame: Une frame de vidéo (image OpenCV)
            detection_threshold: Seuil de confiance pour la détection
        
        Returns:
            Tuple de (frame annotée, informations de détection)
        """
        # Faire une copie de la frame pour l'annotation
        annotated_frame = frame.copy()
        
        # Prétraiter et prédire
        class_id, confidence, sign_name = self.predict(frame)
        
        detection_info = {
            'class_id': class_id,
            'confidence': confidence,
            'sign_name': sign_name
        }
        
        # Annoter la frame si la confiance est supérieure au seuil
        if confidence >= detection_threshold:
            text = f"{sign_name} ({confidence:.2f})"
            cv2.putText(
                annotated_frame, 
                text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 
                2
            )
        
        return annotated_frame, detection_info
    
    def predict_batch(self, images):
        """
        Effectue des prédictions sur un lot d'images
        
        Args:
            images: Liste d'images ou array 4D numpy
            
        Returns:
            Prédictions pour chaque image
        """
        # Prétraiter les images si nécessaire
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_img = preprocess_for_detection(img, self.target_size)
            processed_images.append(processed_img)
        
        # Convertir en array numpy
        batch = np.array(processed_images)
        
        # Prédictions
        predictions = self.model.predict(batch)
        
        # Extraire les classes prédites et les confiances
        results = []
        for pred in predictions:
            class_id = np.argmax(pred)
            confidence = pred[class_id]
            sign_name = self.label_map.get(class_id, 'Unknown') if self.label_map else 'Unknown'
            results.append((class_id, confidence, sign_name))
        
        return results