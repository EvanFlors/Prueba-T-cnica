import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:

  def __init__(self, file_path, test_size=0.2, random_state=42, extension=".csv", stratify_col=None):
    self.ingestion_config = DataIngestionConfig()
    self.file_path = file_path
    self.test_size = test_size
    self.random_state = random_state
    self.stratify_col = stratify_col
    self.extension = extension

  def reduce_large_classes(self, data, label_col, max_samples=10000):
    '''
    Method to reduce large classes before splitting

    Args:
        data: pd.DataFrame
        label_col: str
        max_samples: int

    Returns:
        pd.DataFrame
    '''
    logging.info(f"Reducing classes with more than {max_samples} samples")
    reduced_data = []
    for category, group in data.groupby(label_col):
      if len(group) > max_samples:
        group = group.sample(n = max_samples, random_state = self.random_state)
        logging.info(f"Reducing class {category} to {len(group)} samples")
      reduced_data.append(group)
    return pd.concat(reduced_data).reset_index(drop=True)

  def initiate_data_ingestion(self):
    '''
    Method to initiate data ingestion

    Returns:
        train_data_path: str
        test_data_path: str
    '''

    logging.info("Data ingestion method / component")

    try:
      logging.info(f"Reading data: '{self.file_path}'")
      
      if self.extension == ".csv":
        data = pd.read_csv(self.file_path)
      elif self.extension == ".json":
        data = pd.read_json(self.file_path)
      else:
        raise Exception("Invalid file extension")
      
      data['category'] = data['category'].astype(str)
      data['headline'] = data['headline'].astype(str)
      
      if self.stratify_col:
        logging.info("Reducing large classes before splitting")    
        data = self.reduce_large_classes(data, self.stratify_col)

      os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

      logging.info("Splitting data into train and test")

      stratify_values = data[self.stratify_col] if self.stratify_col is not None else None

      train_set, test_set = train_test_split(
        data,
        test_size=self.test_size,
        random_state=self.random_state,
        stratify=stratify_values
      )

      logging.info(f"Saving train data: '{self.ingestion_config.train_data_path}'")
      train_set.to_csv(self.ingestion_config.train_data_path, index=False)

      logging.info(f"Saving test data: '{self.ingestion_config.test_data_path}'")
      test_set.to_csv(self.ingestion_config.test_data_path, index=False)

      logging.info(f"Saving raw data: '{self.ingestion_config.raw_data_path}'")
      data.to_csv(self.ingestion_config.raw_data_path, index=False)

      logging.info("Ingestion of data is completed")

      return (
        self.ingestion_config.train_data_path,
        self.ingestion_config.test_data_path
      )

    except Exception as e:
      raise CustomException(e, sys)


from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    
    ingeston = DataIngestion(
      file_path = "notebook/data.json",
      test_size = 0.2,
      random_state = 42,
      stratify_col = "category",
      extension = ".json"
    )
    
    train_data_path, test_data_path = ingeston.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)