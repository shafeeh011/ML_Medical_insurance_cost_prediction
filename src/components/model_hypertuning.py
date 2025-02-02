import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass, field
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@dataclass
class ModelConfig:
    """
    Dataclass to hold model configurations and hyperparameter tuning spaces.
    """
    models: dict = field(default_factory=lambda: {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(),
        'K-Neighbors': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor()
    })
    
    param_grids: dict = field(default_factory=lambda: {
        'Random Forest': {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [10, 50, 100, 200],
            'learning_rate': [0.1, 0.05, 0.01],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        },
        'Linear Regression': {
            'fit_intercept': [True, False]
        },
        'SVR': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        },
        'K-Neighbors': {
            'n_neighbors': [3, 5, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'Decision Tree': {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }
    })


class ModelTrainer:
    """
    Class for training and evaluating regression models with hyperparameter tuning.
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

    def hyperparameter_tuning(self, model_name, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV for the best model.

        Parameters:
        model_name (str): Name of the model to tune.
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.

        Returns:
        dict: Best hyperparameters and best score.
        """
        param_grid = self.config.param_grids[model_name]
        grid_search = GridSearchCV(self.config.models[model_name], param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_, -grid_search.best_score_

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

        # Convert y_train and y_test to 1D arrays
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

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

                # Perform hyperparameter tuning for the best performing model
                if test_r2 == max(r2_list):
                    best_params, best_score = self.hyperparameter_tuning(name, X_train, y_train)
                    logging.info(f"Best parameters for {name}: {best_params}")
                    print(f"Best parameters for {name}: {best_params}")
                    print(f"Best score for {name}: {best_score:.4f}")
                    print("=" * 35)
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
