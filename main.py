import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

## load the pre-trained model with relu acticvation
model = load_model('simple_rnn_imdb.h5')

# step 2: Helper functions
#function to decode reviews
def decode_revview(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Step 3: Prediction function
def predict_sentiment(text):
    padded_review = preprocess_text(text)
    prediction = model.predict(padded_review)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]



## Streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative):")
user_input = st.text_area("Review Text", "I love this movie! It's fantastic.")
if st.button("Predict Sentiment"):
    preprocessed_input=preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment= 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")
else:
    st.write("Please enter a movie review.")