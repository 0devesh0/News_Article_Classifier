import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import joblib  # Use joblib for saving/loading
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
# Specify the NLTK data directory
nltk.data.path.append('C:/Users/deves/nltk_data')

# Ensure required nltk packages are downloaded
try:
    nltk.download('punkt', download_dir='C:/Users/deves/nltk_data')
    nltk.download('wordnet', download_dir='C:/Users/deves/nltk_data')
    nltk.download('stopwords', download_dir='C:/Users/deves/nltk_data')
except Exception as e:
    st.error(f"Error downloading NLTK resources: {str(e)}")

# Load your pre-trained Keras model (adjust path as needed)
model = tf.keras.models.load_model("C:/Users/deves/Desktop/news_classifier/bilstm_category_classification_model.h5")

# Load tokenizer using joblib
try:
    tokenizer = joblib.load('tokenizer.pickle')  # Change to joblib for tokenizer
except Exception as e:
    st.error(f"Error loading Tokenizer: {str(e)}")
    tokenizer = None  # Set to None to avoid further errors

# Load label encoder using joblib
try:
    le = joblib.load('label_encoder.pkl')
    # Ensure that le is indeed a LabelEncoder instance
    if not isinstance(le, LabelEncoder):
        raise ValueError("Loaded object is not a LabelEncoder instance.")
except Exception as e:
    st.error(f"Error loading Label Encoder: {str(e)}")
    le = None  # Set to None to avoid further errors

# Print the classes for debugging
if le is not None:
    st.write(f"LabelEncoder classes: {le.classes_}")  # This should work correctly
else:
    st.warning("Label Encoder not loaded correctly.")

# Preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function (same as used during training)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Keep only alphabets
    words = word_tokenize(text)  # Tokenize the text
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
    return ' '.join(words)

# Streamlit app UI
st.title("News Article Classifier")
st.write("Classify news articles based on their headline and short description.")

headline = st.text_input("Enter the headline:")
description = st.text_area("Enter the short description:")

max_length = 50  # Assuming max length used during training

# Function to predict category
def predict_category(text):
    cleaned_text = clean_text(text)  # Apply text cleaning
    sequences = tokenizer.texts_to_sequences([cleaned_text])  # Tokenize text
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')  # Pad sequences
    predictions = model.predict(padded_sequences)  # Make predictions
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability
    
    # Ensure the label encoder is loaded properly before inverse transforming
    if le is not None:
        return le.inverse_transform(predicted_class)[0]  # Convert index to actual category
    else:
        raise ValueError("Label Encoder is not properly loaded.")

# Classify button logic
if st.button("Classify"):
    if headline and description:
        text = headline + " " + description  # Concatenate headline and description
        try:
            category = predict_category(text)  # Predict category
            st.success(f"The predicted category is: {category}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please provide both headline and short description.")
