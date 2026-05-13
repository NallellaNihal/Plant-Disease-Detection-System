import sys
import tensorflow as tf
from src.config import MODEL_PATH
from src.utils import load_class_names, predict_image
from src.recommendations import get_recommendation

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        return

    image_path = sys.argv[1]

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    predicted_class, confidence = predict_image(model, image_path, class_names)

    print("Prediction Result")
    print("-----------------")
    print(f"Disease/Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Recommendation: {get_recommendation(predicted_class)}")

if __name__ == "__main__":
    main()
