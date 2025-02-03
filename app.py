from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

app = FastAPI()

# Define input data structure
class InsuranceInput(BaseModel):
    age: int
    sex: str  # 'male' or 'female'
    bmi: float
    children: int
    smoker: str  # 'yes' or 'no'
    region: str  # 'southeast', 'southwest', 'northeast', 'northwest'

# Load trained model and encoder
insurance_model = pickle.load(open('./models/gradient_boosting_model.pkl', 'rb'))
encoders = pickle.load(open('./models/encoders.pkl', 'rb'))

@app.post('/insurance_prediction')
def insurance_prediction(input_parameters: InsuranceInput):
    try:
        # Convert input data to dictionary
        input_dict = {
            'age': input_parameters.age,
            'sex': input_parameters.sex,
            'bmi': input_parameters.bmi,
            'children': input_parameters.children,
            'smoker': input_parameters.smoker,
            'region': input_parameters.region
        }
        
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Encode categorical variables using preloaded encoders
        for col in ['sex', 'smoker', 'region']:
            input_df[col] = encoders[col].transform(input_df[col])

        # Convert DataFrame to NumPy array
        input_array = input_df.to_numpy()

        # Make prediction
        prediction = insurance_model.predict(input_array)
        return {'predicted_insurance_cost': float(prediction[0])}
    except Exception as e:
        return {'error': str(e)}

# WSGI Entry Point for Elastic Beanstalk
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
