import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

#to run: streamlit run app.py

# Load the serialized SVM model and TF-IDF vectorizer
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Streamlit app interface
st.title('Text Classification based on SVM Model')

# Adding a description
st.write("""
This app uses a Support Vector Machine (SVM) model for classifying text into categories. Simply choose between entering your text or linking the page in the interent in the box below and press 'Classify' to see the predicted category and the model's confidence in its prediction. 
         The app is able to identify the partiuclar text topics: business, entertainment, food, graphics, historical, medical, politics, space, sport, and technology.
""")

st.markdown("""
Should you want to see all models available or use a different dataset for training, 
please refer to this Google Colab notebook:
[Google Colab Notebook](https://colab.research.google.com/drive/1d_RZR8xhiVBPSOCMJKwHvPtWBXE6nAVQ?usp=sharing) or this [Github Repository](https://github.com/Michaelzats).
""")
# Include an option to choose between entering text or providing a URL
mode = st.radio("Choose input mode:", ("Enter text", "Enter URL"))

if mode == "Enter text":
    user_input = st.text_area("Enter text here:", "")
elif mode == "Enter URL":
    user_input = st.text_input("Enter URL here:", "")
    if user_input:
        # Fetch the content from URL
        try:
            response = requests.get(user_input)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text from the HTML soup object
            user_input = soup.get_text(separator=' ')
            # Optionally display the fetched text (you can remove or comment out this line in production)
            st.text_area("Fetched text:", user_input, height=300)
        except Exception as e:
            st.error(f"An error occurred while fetching the article: {e}")

if st.button('Classify'):
    # Ensure the user input is not empty
    if user_input:
        # Transform the input text using the loaded TF-IDF vectorizer
        transformed_input = tfidf_vectorizer.transform([user_input])
        # Predict using the loaded SVM model
        probabilities = svm_model.predict_proba(transformed_input)
        # Get the max probability and the corresponding class index
        max_prob_index = probabilities[0].argmax()
        prediction = svm_model.classes_[max_prob_index]
        confidence = probabilities[0][max_prob_index] * 100  # Convert to percentage
        
        st.write(f'Predicted category: {prediction}')
        st.write(f'Confidence: {confidence:.2f}%')
    else:
        st.write("Please enter some text to classify.")
