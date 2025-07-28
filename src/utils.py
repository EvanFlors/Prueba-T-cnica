import os
import sys
import dill

from sklearn.metrics import accuracy_score, f1_score

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
  '''
  Function to save pickle file
  
  Args:
    file_path (str): path to pickle file
    obj (object): object to be saved in pickle file
  
  Returns:
    None
  '''
  try:
    dir_path = os.path.dirname(file_path)

    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
      dill.dump(obj, file_obj)
      
    logging.info(f"Saved pickle file: {file_path}")

  except Exception as e:
    logging.info("Exception occured in save_object method")
    raise CustomException(e, sys)
  
def evaluate_model(X_train, y_train, X_test, y_test, models, balanced = False):
  '''
  Function to evaluate model performance
  
  Args:
    X_train (pd.DataFrame): training data
    y_train (pd.DataFrame): training labels
    X_test (pd.DataFrame): testing data
    y_test (pd.DataFrame): testing labels
    models (dict): dictionary of models
    balanced (bool): whether to use balanced accuracy or not
    
  Returns:
    report (dict): dictionary of model scores'''
  try:
    report = {}
    
    for i in range(len(list(models))):
      
      model = list(models.values())[i]
      model.fit(X_train, y_train)
      
      y_train_pred = model.predict(X_train)
      y_test_pred = model.predict(X_test)
      
      if not balanced:
        train_model_score = f1_score(y_train, y_train_pred, average = 'weighted')
        test_model_score = f1_score(y_test, y_test_pred, average = 'weighted')
      else:
        train_model_score = accuracy_score(y_train, y_train_pred)
        test_model_score = accuracy_score(y_test, y_test_pred)
      
      report[list(models.keys())[i]] = test_model_score

    return report

  except Exception as e:
    logging.info("Exception occured in evaluate_model method")
    raise CustomException(e, sys)
  
  
def load_object(file_path):
  '''
  Function to load pickle file
  
  Args:
    file_path (str): path to pickle file
    
  Returns:
    obj: object
  '''
  try:
    with open(file_path, "rb") as file_obj:
      return dill.load(file_obj)
    
  except Exception as e:
    logging.info("Exception occured in load_object method")
    raise CustomException(e, sys)
  
print(load_object("artifacts/model.pkl"))