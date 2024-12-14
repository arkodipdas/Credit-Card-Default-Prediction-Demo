import os
import sys
import joblib  # Import joblib for saving and loading models

from src.CreditcardDefaultPrediction.logger import logging
from src.CreditcardDefaultPrediction.exception import CustomException
from src.CreditcardDefaultPrediction.utils.utils import evaluate_model

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier


@dataclass
class ModelTrainingConfig:
    # Updated file extension to .joblib
    trained_model_file_path = os.path.join('artifacts', 'model.joblib')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting independent and dependent features from train and test array")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # Dictionary of models to train
            models = {
                LogisticRegression(): 'Logistic Regression',
                SVC(kernel='rbf', C=10, degree=5, gamma='auto'): "Support Vector",
                KNeighborsClassifier(n_neighbors=20): 'KNN Classifier',
                DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=2, max_features='sqrt', max_depth=8,
                                       criterion='entropy'): 'Decision Tree',
                RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=4, min_samples_leaf=5,
                                       max_features='sqrt'): 'Random Forest',
                GaussianNB(): 'Naive Bayes',
                AdaBoostClassifier(algorithm="SAMME"): 'AdaBoosting',
                GradientBoostingClassifier(): 'Gradient Boosting'
            }

            results = {}

            # Train and evaluate each model
            for model in models.keys():
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                scores = evaluate_model(y_test, y_pred)

                results[scores] = model

            # Find the best model
            best_model_score = max(results.keys())
            best_model = results[best_model_score]
            best_model_name = models[results[best_model_score]]

            print("\n==========================================\n")
            logging.info(f"Best model is {best_model_name} with accuracy of {best_model_score * 100}")
            print(f"Best model is {best_model_name} with accuracy of {best_model_score * 100}")

            # Save the best model using joblib
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)

            return (
                best_model_name,
                best_model_score
            )

        except Exception as e:
            logging.info("An exception has occurred in initiate_model_training")
            raise CustomException(e, sys)
