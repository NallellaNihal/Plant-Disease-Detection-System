import json
import numpy as np
import tensorflow as tf
from PIL import Image
from .config import IMAGE_SIZE, CLASS_NAMES_PATH

def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

def save_class_names(class_names):
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=4)

def preprocess_image(image):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(model, image, class_names):
    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]) * 100)
    predicted_class = class_names[predicted_index]
    return predicted_class, confidence
