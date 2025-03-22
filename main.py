from keras.models import load_model
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

nltk.download("stopwords")
stopwords = list(stopwords.words("english"))

saved_model = load_model("sentiment_analysis.keras")

data = pd.read_csv("EcoPreprocessed.csv")

def create_text_all():
    text_all = ""
    for review in data["review"]:
      text_all += review + " "
    return text_all

def create_train_test_data():
    x = list(data["review"])
    y = list(data["division"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return (x_train, x_test, y_train, y_test) 

def tokenize_text(x):
    text_all = create_text_all()
    text_all = np.array(text_all.split(" "))
    unique_words = np.unique(text_all)
    tokenizer = Tokenizer(num_words = len(unique_words))

x_train, x_test, y_train, y_test = create_train_test_data()

