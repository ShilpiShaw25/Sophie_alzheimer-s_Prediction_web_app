
import streamlit as st
import tensorflow as tf
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import librosa
import librosa.display
import json
import requests
import base64
import io

# Functions
@st.cache_resource
def audio_to_spectrogram(y, sr, save_path):
    # Generate a spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to decibels for better visualization
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.axis('off')  # Remove axes for a clean image
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')

    # Save the spectrogram image
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Save without extra padding
    plt.close()  # Close the plot to free memory

    return save_path

@st.cache_resource
def generate_spectrogram_from_audio(audio_file, saved_image):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Generate and save the spectrogram
    generated_image = audio_to_spectrogram(y, sr, saved_image)

    return generated_image

# Define labels
PREDICTED_LABELS = ["Control", "Alzheimer's"]
PREDICTED_LABELS.sort()


def get_prediction(image_data):
  #replace your image classification ai service URL
  url = 'https://askai.aiclub.world/26c71386-3f71-49a9-b3a8-34e95fb81dbe'  #Edit 
  r = requests.post(url, data=image_data)
  response = r.json()['predicted_label']
  score = r.json()['score']
  #print("Predicted_label: {} and confidence_score: {}".format(response,score))
  return response, score

#title of the web page
st.title("Alzheimer's Prediction")

#setting the main picture
st.image(
    "https://www.psypathy.com/wp-content/uploads/2023/02/Alzheimers-disease-1536x768.jpg ",
    caption = "Alzheimer's")

#about the web app
st.header("About the Web App")
st.write("This web app can predict whether a given audio or speech sample indicates the presence of dementia in a person ")

st.subheader("Audio File Uploader")

# Create file uploader for audio files
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Display the audio file player
    st.audio(uploaded_file, format='audio/wav')

    # Name to save spectrogram
    image_name = "generated_spectrogram.png"

    # Generate and save the spectrogram image from the uploaded audio file
    generated_image = generate_spectrogram_from_audio(uploaded_file, image_name)

    # Convert the image to bytes
    img = Image.open(generated_image).convert('RGB')  # Convert to RGB as it would give erorr --> OSError: cannot write mode RGBA as JPEG 
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    # Convert bytes to base64 encoding
    payload = base64.b64encode(byte_im)

    # Predictions
    response, scores = get_prediction(payload)

    response_label = PREDICTED_LABELS[int(response)]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction Label", response_label)
    with col2:
        st.metric("Confidence Score", max(scores))

else:
    st.write("Please upload an audio file.")
