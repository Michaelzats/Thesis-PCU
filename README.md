# Thesis-PCU
**Thesis Name:**
Automating News Article Classification with Machine Learning Based on Document Classification Aspects

**Projects includes:**
* PART 1 (Google Colab Project Part)
* PART 2 (App Deployment with streamlit)

  
**PART 1 (Google Colab Project Part)**

Google Colab Link ( https://colab.research.google.com/drive/1d_RZR8xhiVBPSOCMJKwHvPtWBXE6nAVQ?usp=sharing )

Description:
This project aims to classify a diverse set of documents into predefined categories using various machine learning and deep learning models.


Installation:
Prerequisites:
* Python 3
* Google Colab (for running the provided notebook)

Installation steps:
* Clone the repository from Google Colab. (https://colab.research.google.com/drive/1j-RytXdZDXCAWKHlaQdFWmRo64SJt_3Y?usp=sharing) In Colab click File --> Save a copy in Colab.
<img width="200" alt="Screenshot 2024-04-01 at 13 16 44" src="https://github.com/Michaelzats/Thesis-PCU/assets/92814061/cfd2e93c-2450-47cb-bdb9-d62b3f967a02">

* Install necessary Python libraries using !pip install
  
!pip install numpy pandas statsmodels scikit-learn matplotlib seaborn nltk gensim tensorflow keras==2.12.0

* Import necessary Python libraries

(import numpy as np
import pandas as pd
import os
from pathlib import Path
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import nltk
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle
from google.colab import files
word2vec_model = api.load("word2vec-google-news-300")

Usage:
Basic examples:
* Load the dataset provided “dataset (10)Dataset Text Document Classification.zip” for part 1 “Dataset BBC Full Text Document Classification.zip” for part 2 and “Data_For_the_App_Training.zip” for part 3 in the project PART 1 in Google Colab. You can do it by clicking on the file folder icon from the right in Colab and then draging “example.zip” File into the opened part.  Example how to drag “DATASETS.zip” into the file folder as an example <img width="173" alt="Screenshot 2024-04-01 at 13 18 06" src="https://github.com/Michaelzats/Thesis-PCU/assets/92814061/e0d7a980-f8c6-4f56-867d-c102ae981ab4">



* Do not unzip it before the excucution, the command # Once uploaded, unzip using: “!unzip example.zip !ls” will do it.
* Train the model using the provided training set.
* Evaluate the model using the test set.

Detailed examples:
* The notebook provides detailed steps for implementing and evaluating each model.

Features:
* Classification of documents into predefined categories.
* Evaluation based on Precision, Recall, F1-Score, and Accuracy.
* Use of various machine learning (Naive Bayes, Support Vector Machines (SVM), Decision Tree and Random Forest) and deep learning model (Deep Neural Network).

Limitations:
* The prototype is recommended to be run in Google Colab as it was created there and can not be simply transferred to the other environment 
* The prototype is computationally heavy, therefore it takes long time to run it through. For the “Dataset BBC Full Text Document Classification.zip” and “Data_For_the_App_Training.zip” datasets execution, google colab pro is required as the datasets are too large and require additional RAM
* The prototype was done on MAC OS environment, therefore should you have any problems with running it on other environments, please contact mikhail.zats@praguecollege.cz

Testing:
* The models are evaluated using the test set provided in the dataset.
* Evaluation metrics include Precision, Recall, F1-Score, and Accuracy.
Contact:
* For any queries or feedback, please reach out to mikhail.zats@praguecollege.cz . 


**PART 2 (App Deployment with streamlit)**

App Link ( https://thesis-pcu-fm4qprwzd3mqu859qynczv.streamlit.app/ )

Installation:
Prerequisites:
Install necessary Python libraries using !pip install:
Usage:
Detailed examples:
Features:
Limitations:
Testing:


Contact Information: 
mikhail.zats@praguecollege.cz
‭+49 173 5923282‬
