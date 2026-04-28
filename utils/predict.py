import tensorflow as tf
import numpy as np

# Modell laden (einmal!)
model = tf.keras.models.load_model("model/keras_model", compile=False)

# Labels laden & bereinigen
with open("model/labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

def predict_image(image):
    # Bild vorbereiten
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    probs = tf.nn.softmax(predictions[0]).numpy()

    index = np.argmax(probs)
    label = labels[index]
    confidence = float(probs[index])

    return label, confidence
