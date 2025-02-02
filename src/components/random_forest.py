import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  # Import the save_object function

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ModelConfig:
    """
    Dataclass to store model hyperparameters.
    """
    learning_rate: float = 0.05
    max_depth: int = 5
    min_samples_leaf: int = 10
    min_samples_split: int = 2
    n_estimators: int = 50
    
    def to_dict(self):
        return {
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'min_samples_split': self.min_samples_split,
            'n_estimators': self.n_estimators
        }

class GradientBoostingModel:
    """
    Class for handling training and evaluation of a Gradient Boosting model.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = GradientBoostingRegressor(**self.config.to_dict())
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model.
        """
        logging.info("Training the model...")
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed successfully.")
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Evaluate the model using Mean Squared Error.
        """
        logging.info("Evaluating the model...")
        try:
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            logging.info(f"Model evaluation completed successfully. MSE: {mse:.4f}")
            return mse
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise
    
    def save_model(self, model_file_path: str):
        """
        Save the trained model to a .pkl file.
        """
        logging.info(f"Saving the trained model to {model_file_path}...")
        try:
            save_object(model_file_path, self.model)
            logging.info(f"Model saved successfully at {model_file_path}")
        except Exception as e:
            logging.error(f"Error saving the model: {e}")
            raise

if __name__ == "__main__":
    try:
        logging.info("Loading training and test datasets...")
        X_train = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_Medical_insurance_cost_prediction/data/data_splits/X_train.csv")
        y_train = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_Medical_insurance_cost_prediction/data/data_splits/y_train.csv").values.ravel()  # Convert to 1D array
        X_test = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_Medical_insurance_cost_prediction/data/data_splits/X_test.csv")
        y_test = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_Medical_insurance_cost_prediction/data/data_splits/y_test.csv").values.ravel()  # Convert to 1D array
        logging.info("Datasets loaded successfully.")

        # Initialize configuration and model
        config = ModelConfig()
        gb_model = GradientBoostingModel(config)
        
        # Train model
        gb_model.train(X_train, y_train)
        
        # Evaluate model
        score = gb_model.evaluate(X_test, y_test)
        logging.info(f"Best score (MSE): {score:.4f}")

        # Save the trained model to a .pkl file
        model_file_path = "/home/muhammed-shafeeh/AI_ML/ML_Medical_insurance_cost_prediction/models/gradient_boosting_model.pkl"
        gb_model.save_model(model_file_path)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
