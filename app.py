import streamlit as st
import tensorflow as tf
from PIL import Image

from src.config import MODEL_PATH
from src.utils import load_class_names, predict_image
from src.recommendations import get_recommendation

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection System")
st.write("Upload a plant leaf image to detect whether it is healthy or diseased.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    if st.button("Predict Disease"):
        try:
            model = load_model()
            class_names = load_class_names()

            predicted_class, confidence = predict_image(model, image, class_names)
            recommendation = get_recommendation(predicted_class)

            st.success(f"Predicted Class: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")
            st.warning(f"Recommendation: {recommendation}")

        except Exception as e:
            st.error("Model not found or not trained yet.")
            st.write("First add dataset images and run:")
            st.code("python train.py")
            st.exception(e)
else:
    st.info("Please upload a leaf image to continue.")
