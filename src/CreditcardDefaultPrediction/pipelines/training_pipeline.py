import os
import sys
from src.CreditcardDefaultPrediction.components.data_transformation import DataTransformation
from src.CreditcardDefaultPrediction.components.model_trainer import ModelTrainer
from src.CreditcardDefaultPrediction.logger import logging
from src.CreditcardDefaultPrediction.exception import CustomException
import pandas as pd

from src.CreditcardDefaultPrediction.components.data_ingestion import DataIngestion


obj = DataIngestion()

train_data_path, test_data_path = obj.initiate_data_ingestion()
data_transformation = DataTransformation()
train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

model_trainer = ModelTrainer()
best_model, best_model_score = model_trainer.initiate_model_training(train_arr, test_arr)