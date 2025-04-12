def predict_image(image_path):
    # Prétraiter l'image
    image = preprocess_image(image_path)
    
    # Ajouter une dimension pour la batch
    image = np.expand_dims(image, axis=0)

    # Prédire
    prediction = model.predict(image)

    # Convertir les prédictions en classe
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class
