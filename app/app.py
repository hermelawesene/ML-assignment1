import streamlit as st
import joblib
import pickle
import pandas as pd
import os
import sys






# Load the trained model
#model = joblib.load('../models/model.pkl')
#model = joblib.load('model.pkl')
modelc = joblib.load('models/model.pkl')

# Title for the web app
st.title('Star Classification Prediction')

# Input fields for each feature
temperature = st.number_input('Temperature (K)', min_value=2000, max_value=40000, value=4500)
luminosity = st.number_input('Luminosity (L/Lo)', min_value=0.0001, max_value=813000.0, value=0.7)
radius = st.number_input('Radius (R/Ro)', min_value=0.0084, max_value=1673.0, value=0.85)
magnitude = st.number_input('Absolute Magnitude (Mv)', min_value=-11.75, max_value=20.0, value=4.7)
star_color = st.selectbox('Star Color', ['Red', 'Blue-White', 'White', 'Yellow-White'])
spectral_class = st.selectbox('Spectral Class', ['A', 'B', 'F', 'G', 'K', 'M', 'O'])

# Prepare the input data for prediction
new_data = pd.DataFrame({
    'Temperature (K)': [temperature],
    'Luminosity(L/Lo)': [luminosity],
    'Radius(R/Ro)': [radius],
    'Absolute magnitude(Mv)': [magnitude],
    'Star color_Blue-White': [1 if star_color == 'Blue-White' else 0],
    'Star color_Red': [1 if star_color == 'Red' else 0],
    'Star color_White': [1 if star_color == 'White' else 0],
    'Star color_Yellow-White': [1 if star_color == 'Yellow-White' else 0],
    'Spectral Class_A': [1 if spectral_class == 'A' else 0],
    'Spectral Class_B': [1 if spectral_class == 'B' else 0],
    'Spectral Class_F': [1 if spectral_class == 'F' else 0],
    'Spectral Class_G': [1 if spectral_class == 'G' else 0],
    'Spectral Class_K': [1 if spectral_class == 'K' else 0],
    'Spectral Class_M': [1 if spectral_class == 'M' else 0],
    'Spectral Class_O': [1 if spectral_class == 'O' else 0]
})

# Align the columns with the model's expected feature names
new_data = new_data[modelc.feature_names_in_]

# Make the prediction
if st.button('Predict Star Type'):
    prediction = modelc.predict(new_data)
    star_type = prediction[0]
    
    # Map the prediction back to the star type
    star_types = {
        0: 'Brown Dwarf',
        1: 'Red Dwarf',
        2: 'White Dwarf',
        3: 'Main Sequence',
        4: 'Supergiants',
        5: 'Hypergiants'
    }
    st.write(f'Predicted Star Type: {star_types[star_type]}')
################################################################################################################
# Load the trained model app

# model = joblib.load('/models/regressionmodel.pkl')
# Define the model path
# Append parent directory to sys.path
#sys.path.append(os.path.join(os.path.abspath('.')))
#model_path = 'models/regressionmodel.pkl'
model_path = 'models/linear_regression_model.pkl'

# Try loading the model
try:
    model = pickle.load(open(model_path, "rb"))
    st.success("Model successfully loaded!")
except FileNotFoundError:
    st.error(f"Model not found at {model_path}. Please check the path.")

# Set up the Streamlit interface
st.title("Car Price Prediction")

st.write("""
This app predicts the price of a car based on the following features:
- Levy
- Model
- Prod. year
- Category
- Leather interior
- Engine volume
- Mileage
- Cylinders
- Airbags
""")

# Create input fields for each feature
levy = st.number_input("Levy", min_value=0, max_value=10000, step=1)
model_input = st.selectbox("Model", ['Model A', 'Model B', 'Model C'])  # Add actual models here
prod_year = st.number_input("Production Year", min_value=1900, max_value=2023, step=1)
category = st.selectbox("Category", ['Sedan', 'SUV', 'Truck'])  # Add categories
leather_interior = st.selectbox("Leather Interior", ['Yes', 'No'])
engine_volume = st.number_input("Engine Volume (L)", min_value=0.5, max_value=10.0, step=0.1)
mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, step=1000)
cylinders = st.number_input("Cylinders", min_value=1, max_value=16, step=1)
airbags = st.number_input("Airbags", min_value=1, max_value=10, step=1)

# Convert categorical features to integers using the same encoding
model_encoder = {'Model A': 0, 'Model B': 1, 'Model C': 2}  # Adjust based on your encoding
category_encoder = {'Sedan': 0, 'SUV': 1, 'Truck': 2}  # Adjust based on your encoding
leather_interior_encoder = {'Yes': 1, 'No': 0}  # Adjust based on your encoding

# Prepare the input features as a DataFrame
input_data = pd.DataFrame({
    'Levy': [levy],
    'Model': [model_encoder[model_input]],
    'Prod. year': [prod_year],
    'Category': [category_encoder[category]],
    'Leather interior': [leather_interior_encoder[leather_interior]],
    'Engine volume': [engine_volume],
    'Mileage': [mileage],
    'Cylinders': [cylinders],
    'Airbags': [airbags]
})

# Make predictions using the trained model
predicted_price = model.predict(input_data)

# Access the scalar value from the numpy array and format it correctly
st.write(f"Predicted Car Price: ${predicted_price[0][0]:,.2f}")


##########################################################################################################################################
