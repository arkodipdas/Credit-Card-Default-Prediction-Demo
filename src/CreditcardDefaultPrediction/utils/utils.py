import os
import sys
from joblib import dump, load  # Import joblib for saving and loading objects

from src.CreditcardDefaultPrediction.logger import logging
from src.CreditcardDefaultPrediction.exception import CustomException

from sklearn.metrics import accuracy_score

def save_object(file_path, obj):
    """
    Saves a Python object to the specified file path using joblib.
    Creates the directory if it does not exist.
    """
    try:
        dir_path = os.path.dirname(file_path)  # Extract directory path from file path
        os.makedirs(dir_path, exist_ok=True)   # Create the directory if it doesn't exist

        dump(obj, file_path)  # Save the object using joblib

    except Exception as e:
        logging.info("Exception occurred in save_object method")
        raise CustomException(e, sys)


def evaluate_model(y_true, y_pred):
    """
    Evaluates a model's performance by calculating accuracy score.
    """
    try:
        return accuracy_score(y_true, y_pred)  # Calculate accuracy score

    except Exception as e:
        logging.info("Exception occurred in evaluate_model function")
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a Python object from the specified file path using joblib.
    """
    try:
        return load(file_path)  # Load the object using joblib

    except Exception as e:
        logging.info("Exception occurred in loading object")
        raise CustomException(e, sys)
