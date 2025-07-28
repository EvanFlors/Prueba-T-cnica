import sys
import re
import string
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

class CustomData:
    def __init__(self, headline: str):
        self.headline = headline

    def get_data_as_dataframe(self):
        try:
            clean = preprocess_text(self.headline)
            char_count = len(self.headline)
            word_count = len(self.headline.split())
            avg_word_len = char_count / word_count if word_count > 0 else 0
            punctuation_count = sum(1 for c in self.headline if c in string.punctuation)
            upper_case_count = sum(1 for c in self.headline if c.isupper())
            upper_case_ratio = upper_case_count / char_count if char_count > 0 else 0
            starts_with_number = int(self.headline[0].isdigit()) if self.headline else 0
            contains_number = int(any(c.isdigit() for c in self.headline))

            input_dict = {
                "headline": [self.headline],
                "clean_headline": [clean],
                "word_count": [word_count],
                "avg_word_len": [avg_word_len],
                "punctuation_count": [punctuation_count],
                "upper_case_ratio": [upper_case_ratio],
                "starts_with_number": [starts_with_number],
                "contains_number": [contains_number],
            }

            return pd.DataFrame(input_dict)

        except Exception as e:
            raise CustomException(e, sys)

class PredictPipeline:
    def __init__(self):
        self.vectorizer_path = 'artifacts/bow_vectorizer.pkl'
        self.scaler_path = 'artifacts/scaler.pkl'
        self.model_path = 'artifacts/model.pkl'
        self.label_encoder_path = 'artifacts/label_encoder.pkl'

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Loading model and preprocessing objects...")
            vectorizer = load_object(self.vectorizer_path)
            scaler = load_object(self.scaler_path)
            model = load_object(self.model_path)
            label_encoder = load_object(self.label_encoder_path)

            logging.info("Transforming clean headline with BoW...")
            X_text = vectorizer.transform(features['clean_headline'])

            logging.info("Extracting and scaling numerical features...")
            numerical_features = features[['word_count', 'avg_word_len', 'punctuation_count',
                                          'upper_case_ratio', 'starts_with_number', 'contains_number']].values
            numerical_scaled = scaler.transform(numerical_features)

            logging.info("Concatenating text and numerical features...")
            from scipy.sparse import hstack
            final_input = hstack([X_text, numerical_scaled])

            prediction = model.predict(final_input)
            label = label_encoder.inverse_transform(prediction)
            return label[0]

        except Exception as e:
            raise CustomException(e, sys)
