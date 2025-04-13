import cv2
import numpy as np
import pickle
from keras.models import load_model

# Paramètres
frame_width = 640
frame_height = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

# Charger le modèle
model = load_model("models/traffic_sign_model.h5")

# Fonction pour prétraiter l'image
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# Dictionnaire des classes (adapté à votre CSV)
class_names = {
    0: 'Limitation de vitesse (20km/h)',
    1: 'Limitation de vitesse (30km/h)',
    # ... (ajoutez toutes les classes comme dans votre CSV)
}

# Capture vidéo
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, brightness)

while True:
    success, img_original = cap.read()
    img = np.asarray(img_original)
    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    img = img.reshape(1, 32, 32, 1)

    # Prédiction
    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    probability = np.amax(predictions)

    if probability > threshold:
        cv2.putText(img_original, f"CLASS: {class_id} {class_names.get(class_id, 'Unknown')}", (20, 35), font, 0.75, (0, 0, 255), 2)
        cv2.putText(img_original, f"PROBABILITY: {probability * 100:.2f}%", (20, 75), font, 0.75, (0, 0, 255), 2)

    cv2.imshow("Result", img_original)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()