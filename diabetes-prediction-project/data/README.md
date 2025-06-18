### Step 1: Install Required Libraries
Make sure you have `joblib` and `streamlit` installed. You can install them using pip if you haven't already:

```bash
!pip install joblib streamlit
```

### Step 2: Add the New Cell
Add the following code in a new cell in your Jupyter notebook:

```python
import joblib

# Save the best model (assuming it's the last model in the tuned_models dictionary)
best_model_name = eval_models.iloc[0]['Model']  # Get the name of the best model
best_model = tuned_models[best_model_name]  # Retrieve the best model from the tuned_models dictionary

# Save the model as a .pkl file
model_filename = f"{best_model_name.replace(' ', '_')}_model.pkl"
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")

# Streamlit application code
streamlit_code = f"""
import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('{model_filename}')

# Title of the app
st.title('Diabetes Prediction App')

# Input features
st.header('Input Features')
genhlth = st.selectbox('General Health (1-5)', [1, 2, 3, 4, 5])
highbp = st.selectbox('High Blood Pressure (0 or 1)', [0, 1])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
highchol = st.selectbox('High Cholesterol (0 or 1)', [0, 1])
sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
income = st.selectbox('Income Level (1-5)', [1, 2, 3, 4, 5])

# Create a DataFrame for the input features
input_data = pd.DataFrame({
    'GenHlth': [genhlth],
    'HighBP': [highbp],
    'Age': [age],
    'BMI': [bmi],
    'HighChol': [highchol],
    'Sex': [sex],
    'Income': [income]
})

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write('Prediction:', 'Diabetes' if prediction[0] == 1 else 'No Diabetes')
"""

# Save the Streamlit code to a .py file
with open('diabetes_prediction_app.py', 'w') as f:
    f.write(streamlit_code)

print("Streamlit application code saved as 'diabetes_prediction_app.py'.")
```

### Explanation of the Code:
1. **Model Saving**: The code saves the best model from the `tuned_models` dictionary using `joblib.dump()`.
2. **Streamlit Application**: It creates a simple Streamlit application that allows users to input features and get predictions. The application loads the saved model and uses it to make predictions based on user input.
3. **File Saving**: The Streamlit code is saved to a file named `diabetes_prediction_app.py`.

### Step 3: Running the Streamlit Application
To run the Streamlit application, navigate to the directory where the `diabetes_prediction_app.py` file is saved and run the following command in your terminal:

```bash
streamlit run diabetes_prediction_app.py
```

This will start a local server, and you can access the application in your web browser.