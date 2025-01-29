import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelConfig:
    """
    Dataclass to hold model configurations.
    """
    models: dict = field(default_factory=lambda: {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBRegressor": XGBRegressor(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor": AdaBoostRegressor(),
    })


class ModelTrainer:
    """
    Class for training and evaluating regression models.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        logging.info("ModelTrainer initialized with configuration: %s", self.config)

    @staticmethod
    def evaluate_model(true, predicted):
        """
        Evaluate model performance using regression metrics.

        Parameters:
        true (np.ndarray): True target values.
        predicted (np.ndarray): Predicted target values.

        Returns:
        tuple: MAE, RMSE, and R2 score.
        """
        mae = mean_absolute_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate models in the configuration.

        Parameters:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        X_test (pd.DataFrame): Testing feature data.
        y_test (pd.Series): Testing target data.

        Returns:
        None: Prints performance metrics for each model.
        """
        model_list = []
        r2_list = []

        for name, model in self.config.models.items():
            logging.info(f"Training and evaluating model: {name}")
            try:
                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Evaluate model on training and test data
                train_mae, train_rmse, train_r2 = self.evaluate_model(y_train, y_train_pred)
                test_mae, test_rmse, test_r2 = self.evaluate_model(y_test, y_test_pred)

                # Log and print results
                model_list.append(name)
                r2_list.append(test_r2)

                print(f"{name}")
                print("Model performance for Training set")
                print(f"- Root Mean Squared Error: {train_rmse:.4f}")
                print(f"- Mean Absolute Error: {train_mae:.4f}")
                print(f"- R2 Score: {train_r2:.4f}")
                print("----------------------------------")
                print("Model performance for Test set")
                print(f"- Root Mean Squared Error: {test_rmse:.4f}")
                print(f"- Mean Absolute Error: {test_mae:.4f}")
                print(f"- R2 Score: {test_r2:.4f}")
                print("=" * 35)
                print("\n")

            except Exception as e:
                logging.error(f"Error occurred while training {name}: {e}")
                print(f"Error occurred while training {name}: {e}")

        # Summarize best models
        best_model_index = np.argmax(r2_list)
        print(f"Best Model: {model_list[best_model_index]} with R2 Score: {r2_list[best_model_index]:.4f}")


if __name__ == "__main__":
    try:
        # Load the data splits
        X_train = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_Medical_insurance_cost_prediction/data/data_splits/X_train.csv")
        y_train = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_Medical_insurance_cost_prediction/data/data_splits/y_train.csv")
        X_test = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_Medical_insurance_cost_prediction/data/data_splits/X_test.csv")
        y_test = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_Medical_insurance_cost_prediction/data/data_splits/y_test.csv")

        # Initialize ModelConfig and ModelTrainer
        config = ModelConfig()
        trainer = ModelTrainer(config)

        # Train and evaluate models
        trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
