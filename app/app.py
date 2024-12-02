import streamlit as st
import joblib
import pandas as pd

# Load the trained model
#model = joblib.load('../models/model.pkl')
#model = joblib.load('model.pkl')
model = joblib.load('models/model.pkl')

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
new_data = new_data[model.feature_names_in_]

# Make the prediction
if st.button('Predict Star Type'):
    prediction = model.predict(new_data)
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
