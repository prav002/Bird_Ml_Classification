import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import os

# Paths to your files
MODEL_JSON_PATH = r'E:\bird\models\bird_model (1).json'  # Use raw string to avoid escape issues
MODEL_WEIGHTS_PATH = r'E:\bird\models\bird_model_weights.weights (1).h5'  # Use raw string to avoid escape issues
BIRD_SPECIES_CSV_PATH = r'E:\bird\updated_train (1).csv'  # Use raw string to avoid escape issues

# Load Keras model
try:
    with open(MODEL_JSON_PATH, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(MODEL_WEIGHTS_PATH)
    loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    st.write("Keras model loaded and compiled successfully.")
except Exception as e:
    st.write(f"Error loading the Keras model: {e}")

# Load bird species mapping
try:
    bird_species_mapping = pd.read_csv(BIRD_SPECIES_CSV_PATH)[['class', 'bird_species']].set_index('class')['bird_species'].to_dict()
except Exception as e:
    st.write(f"Error loading bird species mapping: {e}")

# Streamlit UI
st.title("Bird Species Predictor")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Debugging: Check image shape
    st.write(f"Image shape: {img_array.shape}")

    # Prediction
    predictions = loaded_model.predict(img_array)
    
    # Debugging: Check predictions
    st.write(f"Predictions: {predictions}")

    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions)
    bird_name = bird_species_mapping.get(predicted_class, "Unknown species")

    st.write(f"Predicted species: {bird_name} (Confidence: {confidence_score:.2f})")
