import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Function for Medical Insurance Prediction
def insurance_prediction(input_data):
    # Load the trained model and encoders
    loaded_model = pickle.load(open('./models/gradient_boosting_model.pkl', 'rb'))
    encoders = pickle.load(open('./models/encoders.pkl', 'rb'))

    # Convert input data to a dictionary
    input_dict = {
        'age': int(input_data[0]),
        'sex': input_data[1],
        'bmi': float(input_data[2]),
        'children': int(input_data[3]),
        'smoker': input_data[4],
        'region': input_data[5]
    }

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Encode categorical variables using pre-trained encoders
    for col in ['sex', 'smoker', 'region']:
        input_df[col] = encoders[col].transform(input_df[col])

    # Convert DataFrame to NumPy array
    input_array = input_df.to_numpy()

    # Make prediction
    prediction = loaded_model.predict(input_array)
    
    return f'Estimated Insurance Cost: ${prediction[0]:,.2f}'


# Streamlit Web App
def main():
    # Title
    st.title('Medical Insurance Cost Prediction')

    # User Input Fields
    age = st.text_input('Age')
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.text_input('BMI')
    children = st.text_input('Number of Children')
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    region = st.selectbox('Region', ['southeast', 'southwest', 'northeast', 'northwest'])

    # Prediction Button
    result = ''
    if st.button('Predict Insurance Cost'):
        result = insurance_prediction([age, sex, bmi, children, smoker, region])

    # Display Result
    st.success(result)


# Run the Streamlit App
if __name__ == '__main__':
    main()
