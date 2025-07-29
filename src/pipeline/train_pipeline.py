from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

class TrainPipeline:
  def __init__(self):
    self.data_ingestion = DataIngestion(file_path = "notebook/data.json", test_size = 0.2, random_state = 42, stratify_col = "category", extension = ".json")
    self.data_transformation = DataTransformation()
    self.model_trainer = ModelTrainer()

  def run(self):
    '''
    Method to run the training pipeline
    
    Returns:
        None
    '''
    train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
    X_train, X_test, y_train, y_test = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    if self.model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test):
      logging.critical("Model training completed")
    
if __name__ == "__main__":
  train_pipeline = TrainPipeline()
  train_pipeline.run()