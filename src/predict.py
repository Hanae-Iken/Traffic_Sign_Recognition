import cv2
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Paramètres
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.8  # Seuil augmenté pour plus de fiabilité
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 255, 0)  # Vert pour meilleure visibilité
TEXT_THICKNESS = 2

def load_model_and_labels():
    """Charge le modèle et les labels"""
    model = load_model("models/traffic_sign_model.keras")
    with open("models/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    return model, labels

def preprocess_image(image):
    """Prétraitement optimisé de l'image"""
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def main():
    model, class_names = load_model_and_labels()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = preprocess_image(frame)
        preds = model.predict(processed, verbose=0)[0]
        (label, prob) = (np.argmax(preds), np.max(preds))
        
        if prob > CONFIDENCE_THRESHOLD:
            text = f"{class_names[label]} ({prob*100:.1f}%)"
            cv2.putText(frame, text, (20, 40), FONT, 0.8, TEXT_COLOR, TEXT_THICKNESS)
        
        cv2.imshow("Traffic Sign Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()