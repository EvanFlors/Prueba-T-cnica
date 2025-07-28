import sys

from src.pipeline.train_pipeline import TrainPipeline
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
  try:
    train_pipeline = TrainPipeline()
    train_pipeline.run()
  except Exception as e:
    raise CustomException(e, sys)