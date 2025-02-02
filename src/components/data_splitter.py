import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging  # Assuming you want to use this custom logger

@dataclass
class DataSplitter:
    df: pd.DataFrame
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    save_directory: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    save_dir: str = os.path.join(save_directory, "data/data_splits")
    encoder_file: str = os.path.join(save_directory, "models/encoders.pkl")

    def encode_categorical_data(self):
        """
        Encodes categorical data into numeric format using Label Encoding for simplicity.
        Also saves the encoders as a pickle file.
        
        Returns:
        pd.DataFrame: DataFrame with categorical columns encoded.
        """
        logging.info("Starting categorical data encoding process.")
        
        try:
            categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
            categorical_columns = [col for col in categorical_columns if col != self.target_column]
            
            label_encoders = {}
            for col in categorical_columns:
                logging.info(f"Encoding column: {col}")
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                label_encoders[col] = le
            
            # Save the encoders
            with open(self.encoder_file, 'wb') as f:
                pickle.dump(label_encoders, f)
            
            logging.info(f"Categorical data encoding completed. Encoders saved at {self.encoder_file}")
            return self.df, label_encoders
        except Exception as e:
            raise CustomException(f"An error occurred during categorical data encoding: {e}")

    def split_and_save(self):
        """
        Splits the dataset into training and testing sets and saves the splits as CSV files.
        
        Returns:
        X_train, X_test, y_train, y_test: The training and testing sets.
        """
        logging.info("Starting data splitting process.")
        
        try:
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            logging.info("Data split into training and testing sets.")
            
            os.makedirs(self.save_dir, exist_ok=True)
            
            X_train.to_csv(os.path.join(self.save_dir, "X_train.csv"), index=False)
            X_test.to_csv(os.path.join(self.save_dir, "X_test.csv"), index=False)
            y_train.to_csv(os.path.join(self.save_dir, "y_train.csv"), index=False)
            y_test.to_csv(os.path.join(self.save_dir, "y_test.csv"), index=False)
            logging.info(f"Data splits saved in directory: {self.save_dir}")
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(f"An error occurred during data splitting: {e}")

# Example Usage
if __name__ == "__main__":
    df = pd.read_csv(
        "/home/muhammed-shafeeh/AI_ML/ML_Medical_insurance_cost_prediction/data/kaggle_data/insurance.csv"
    )
    splitter = DataSplitter(df=df, target_column="charges")
    df_encoded, label_encoders = splitter.encode_categorical_data()
    splitter.split_and_save()
