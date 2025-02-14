import json
import requests

# Update the API endpoint for insurance prediction
#url = 'http://0.0.0.0:8000/insurance_prediction'
url = 'http://127.0.0.1:8000/insurance_prediction'

# Example input data for insurance cost prediction
input_data = {
    'age': 33,
    'sex': 'male',
    'bmi': 28.5,
    'children': 2,
    'smoker': 'no',
    'region': 'southeast'
}

# Convert the input dictionary to JSON format
input_json = json.dumps(input_data)

# Send a POST request to the API
response = requests.post(url, data=input_json)

# Print the response from the API
print(response.text)
