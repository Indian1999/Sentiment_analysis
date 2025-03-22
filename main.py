from keras.models import load_model
import nltk
from nltk.corpus import stopwords
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