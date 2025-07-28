import sys
import os
import re
import string
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

category_group_map = {
    'POLITICS': 'POLITICS',
    'U.S. NEWS': 'NEWS',
    'WORLD NEWS': 'NEWS',
    'THE WORLDPOST': 'NEWS',
    'WORLDPOST': 'NEWS',
    'CRIME': 'SOCIETY & EVENTS',
    'WEIRD NEWS': 'SOCIETY & EVENTS',
    'GOOD NEWS': 'SOCIETY & EVENTS',
    'IMPACT': 'SOCIETY & EVENTS',
    'SOCIAL IMPACT': 'SOCIETY & EVENTS',
    'ENTERTAINMENT': 'ENTERTAINMENT & MEDIA',
    'COMEDY': 'ENTERTAINMENT & MEDIA',
    'MEDIA': 'ENTERTAINMENT & MEDIA',
    'SCIENCE': 'SCIENCE & TECH',
    'TECH': 'SCIENCE & TECH',
    'EDUCATION': 'SCIENCE & TECH',
    'COLLEGE': 'SCIENCE & TECH',
    'CULTURE & ARTS': 'ART & STYLE',
    'ARTS & CULTURE': 'ART & STYLE',
    'ARTS': 'ART & STYLE',
    'STYLE & BEAUTY': 'ART & STYLE',
    'STYLE': 'ART & STYLE',
    'WELLNESS': 'HEALTH & WELLNESS',
    'HEALTHY LIVING': 'HEALTH & WELLNESS',
    'FOOD & DRINK': 'FOOD',
    'TASTE': 'FOOD',
    'WOMEN': 'DIVERSITY',
    'BLACK VOICES': 'DIVERSITY',
    'QUEER VOICES': 'DIVERSITY',
    'LATINO VOICES': 'DIVERSITY',
    'HOME & LIVING': 'LIFESTYLE',
    'PARENTING': 'LIFESTYLE',
    'PARENTS': 'LIFESTYLE',
    'TRAVEL': 'LIFESTYLE',
    'RELIGION': 'LIFESTYLE',
    'WEDDINGS': 'LIFESTYLE',
    'DIVORCE': 'LIFESTYLE',
    'FIFTY': 'LIFESTYLE',
    'BUSINESS': 'BUSINESS',
    'MONEY': 'BUSINESS',
    'GREEN': 'ENVIRONMENT',
    'ENVIRONMENT': 'ENVIRONMENT',
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

@dataclass
class DataTransformationConfig:
    label_encoder_path = os.path.join('artifacts', 'label_encoder.pkl')
    bow_vectorizer_path = os.path.join('artifacts', 'bow_vectorizer.pkl')

class DataTransformation:

    def __init__(self):
      self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
      '''
      Function to clean data
      
      Args:
        data (pd.DataFrame): dataframe to be cleaned
      
      Returns:
        data (pd.DataFrame): cleaned dataframe
      '''
      try:
        logging.info("Starting data cleaning...")

        data = data.drop_duplicates(subset="headline")
        data = data[data['headline'].str.strip() != '']
        logging.info(f"Removed duplicates and empty headlines. Data shape: {data.shape}")

        data['category'] = data['category'].replace(category_group_map)

        category_counts = data['category'].value_counts()
        valid_categories = category_counts[category_counts >= 100].index
        data = data[data['category'].isin(valid_categories)]
        logging.info(f"Filtered categories with ≥100 samples. Data shape: {data.shape}")

        data['clean_headline'] = data['headline'].apply(preprocess_text)
        data = data[data['clean_headline'].str.strip() != '']

        data['upper_case_count'] = data['headline'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
        data['char_count'] = data['headline'].apply(lambda x: len(str(x)))
        data['word_count'] = data['headline'].apply(lambda x: len(str(x).split()))
        data['avg_word_len'] = data.apply(lambda x: x['char_count'] / x['word_count'] if x['word_count'] > 0 else 0, axis=1)
        data['punctuation_count'] = data['headline'].apply(lambda x: sum(1 for c in str(x) if c in string.punctuation))
        data['upper_case_ratio'] = data.apply(lambda x: x['upper_case_count'] / x['char_count'] if x['char_count'] > 0 else 0, axis=1)
        data['starts_with_number'] = data['headline'].apply(lambda x: str(x)[0].isdigit()).astype(int)
        data['contains_number'] = data['headline'].apply(lambda x: any(c.isdigit() for c in str(x))).astype(int)

        logging.info("Data cleaning and feature engineering completed successfully.")
        return data

      except Exception as e:
        logging.error("Error during data_cleaning: %s", str(e))
        raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
      '''
      Function to initiate data transformation
      
      Args:
        train_path (str): path to train dataset
        test_path (str): path to test dataset
      
      Returns:
        None
      '''
      try:
        logging.info("Loading train and test datasets...")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        original_train_len = len(train_data) 

        data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        logging.info(f"Concatenated dataset shape: {data.shape}")

        data = self.data_cleaning(data)
        
        le = LabelEncoder()
        data['category'] = le.fit_transform(data['category'])

        train_data = data.iloc[:original_train_len].reset_index(drop=True)
        test_data = data.iloc[original_train_len:].reset_index(drop=True)
        
        print(test_data.head())
        logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

        vectorizer = CountVectorizer(max_features=5000)
        X_train_text = vectorizer.fit_transform(train_data['clean_headline']).toarray()
        X_test_text = vectorizer.transform(test_data['clean_headline']).toarray()
        
        logging.info("BoW vectorization completed.")

        feature_cols = [
          'word_count', 'avg_word_len', 'punctuation_count',
          'upper_case_ratio', 'starts_with_number', 'contains_number'
        ]
        
        X_train_num = train_data[feature_cols].values
        X_test_num = test_data[feature_cols].values

        X_train = np.hstack((X_train_text, X_train_num))
        X_test = np.hstack((X_test_text, X_test_num))
        
        y_train = train_data['category'].values
        y_test = test_data['category'].values

        save_object(self.data_transformation_config.bow_vectorizer_path, vectorizer)
        logging.info("BoW vectorizer saved successfully.")
        
        save_object(self.data_transformation_config.label_encoder_path, le)
        logging.info("Label encoder saved successfully.")

        return X_train, X_test, y_train, y_test

      except Exception as e:
        logging.error("Error during initiate_data_transformation: %s", str(e))
        raise CustomException(e, sys)

if __name__ == '__main__':
  obj = DataTransformation()
  X_train, X_test, y_train, y_test = obj.initiate_data_transformation(train_path='artifacts/train.csv', test_path='artifacts/test.csv')
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)