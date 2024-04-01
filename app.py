import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup

# to run: streamlit run app.py

# Load the serialized SVM model and TF-IDF vectorizer
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Streamlit app interface setup
st.title('Text Classification based on SVM Model')

# Description and links
st.markdown('---')
st.write("""
This app uses a Support Vector Machine (SVM) model for classifying text from articles in English language into categories. Simply choose between entering your text, uploading a text file, or linking the article page in the internet with the URL* in the box below and press 'Classify' to see the predicted category.
""")
st.write("""
The app is able to identify particular text topics: business, entertainment, politics, sport, and technology. The app is developed as part of the Master Thesis work of Michael Zats.
""")
st.markdown("""
Should you want to see all models available or use a different dataset for training, 
please refer to this [Google Colab Notebook](https://colab.research.google.com/drive/1d_RZR8xhiVBPSOCMJKwHvPtWBXE6nAVQ?usp=sharing) or this [Github Repository](https://github.com/Michaelzats/Thesis-PCU).
""")

st.markdown("""
Linking URL Pages option*: sometimes this option may not work as some particular websites do not let the crawler to go through.
""")
st.markdown('---')

# Choose input mode
mode = st.radio("Choose input mode:", ("Enter text", "Upload text file", "Enter URL"))

if mode == "Enter text":
    user_input = st.text_area("Enter text here:", "")
elif mode == "Upload text file":
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    if uploaded_file is not None:
        user_input = str(uploaded_file.read(), "utf-8")  # Convert to string
        st.text_area("File Contents:", user_input, height=300)
elif mode == "Enter URL":
    user_input = st.text_input("Enter URL here:", "")
    if user_input:
        try:
            response = requests.get(user_input)
            soup = BeautifulSoup(response.content, 'html.parser')
            user_input = soup.get_text(separator=' ')
            st.text_area("Fetched text:", user_input, height=300)
        except Exception as e:
            st.error(f"An error occurred while fetching the article: {e}")

if st.button('Classify'):
    if user_input:
        # Transform and classify the input text
        transformed_input = tfidf_vectorizer.transform([user_input])
        probabilities = svm_model.predict_proba(transformed_input)
        max_prob_index = probabilities[0].argmax()
        prediction = svm_model.classes_[max_prob_index]

        st.subheader('Classification Result')
        st.write(f'Predicted category: **{prediction}**')
    else:
        st.warning("Please enter some text, upload a file, or enter a URL to classify.")
