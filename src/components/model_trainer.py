import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
  trained_model_path = os.path.join('artifacts', 'model.pkl')
  
  
class ModelTrainer:
  
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
    
  def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
    '''
    Function to initiate model training
    
    Args:
      X_train (pd.DataFrame): training data
      X_test (pd.DataFrame): testing data
      y_train (pd.DataFrame): training labels
      y_test (pd.DataFrame): testing labels
      
    Returns:
      None
    '''
    try:
      logging.info("Entered initiate_model_trainer method of ModelTrainer class")
      
      models = {
        'Logistic Regression': LogisticRegression(max_iter = 1000, random_state = 42, n_jobs = -1),
        
        # Descomentar para evaluar otros modelos
        
        # 'Random Forest': RandomForestClassifier(max_depth = None, random_state = 42, n_estimators = 100, n_jobs = -1),
        # 'Naive Bayes': GaussianNB(),
        # 'SVM': SVC(max_iter = 1000, random_state = 42, probability = True)
      }
      
      logging.info("Training models: {}".format(list(models.keys())))
      
      model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
      best_model_score = max(sorted(model_report.values()))
      
      #if best_model_score < 0.6:
      #  raise Exception("No suitable model found.")
          
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
      best_model = models[best_model_name]
      logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
      
      save_object(
        file_path = self.model_trainer_config.trained_model_path,
        obj = best_model
      )
      
      logging.info(f"Saved model at {self.model_trainer_config.trained_model_path}")
      
      return True
  
    except Exception as e:
      logging.info("Exception occured in initiate_model_trainer method")
      raise CustomException(e, sys)