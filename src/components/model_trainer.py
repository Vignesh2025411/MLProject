import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42, verbosity=0),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            logging.info("Evaluating all models...")
            # ✅ FIXED: Not passing params (set to None by default in evaluate_models)
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # Log all model performances
            logging.info("Model Performance Report:")
            for model_name, scores in model_report.items():
                logging.info(f"{model_name} - Train: {scores['train']:.4f}, Test: {scores['test']:.4f}")

            # Find best model by test score
            best_model_name = max(
                model_report.keys(),
                key=lambda x: model_report[x]['test']
            )
            best_model_score = model_report[best_model_name]['test']
            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with R² = {best_model_score:.4f}")

            # ✅ IMPROVED: Better threshold check
            if best_model_score < 0.3:
                raise CustomException(
                    f"No best model found. Best score: {best_model_score:.4f} (too low)"
                )

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved successfully")

            return best_model_score

        except Exception as e:
            raise CustomException(str(e), sys)