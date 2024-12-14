import os
import sys
import pandas as pd 
import numpy as np 
from src.CreditcardDefaultPrediction.logger import logging
from src.CreditcardDefaultPrediction.exception import CustomException 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path



@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join('artifacts', 'train.csv')
    test_path: str = os.path.join('artifacts', 'test.csv')
    raw_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            df = pd.read_csv(Path(os.path.join('notebooks/data', 'UCI_Credit_Card.csv')))
            logging.info("I have read the dataset")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_path)), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_path, index=False)

            logging.info("I performed train, test, split")

            
            train_set, test_set = train_test_split(df,test_size = 0.2, random_state=30)
            logging.info("Train,Test,Split completed")


            train_set.to_csv(self.ingestion_config.train_path, index = False,header=True)
            test_set.to_csv(self.ingestion_config.test_path, index = False,header=True)

            logging.info("Data ingestion completed")


            return (
                
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
                
            )





        except Exception as e:
            logging.info("An exception has occurred in Data ingestion method.")
            raise CustomException(e, sys)