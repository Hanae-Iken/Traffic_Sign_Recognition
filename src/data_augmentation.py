import cv2
import numpy as np

def augment_data(image):
    # Rotation aléatoire de l'image
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)  # Rotation de 15 degrés
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # Translation aléatoire
    M_translation = np.float32([[1, 0, 50], [0, 1, 50]])  # Décalage de 50 pixels
    translated_image = cv2.warpAffine(rotated_image, M_translation, (cols, rows))

    # Zoom
    zoomed_image = cv2.resize(translated_image, None, fx=1.1, fy=1.1)  # Zoom avant de 10%

    return zoomed_image
