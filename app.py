# Example Streamlit App for Flight Price Prediction
import streamlit as st
import joblib

# Load the model
model = joblib.load('flight_price_predictor.pkl')

# Create widgets for user input
airline = st.selectbox('Airline', ['Jet Airways', 'SpiceJet', ...])  # Add other airline names
source = st.selectbox('Source', ['Bangalore', ...])  # Add other source locations
destination = st.selectbox('Destination', ['New Delhi', ...])  # Add other destination locations
dep_hour = st.slider('Departure Hour', 0, 23, 12)
dep_min = st.slider('Departure Minute', 0, 59, 30)
duration_minutes = st.slider('Duration (Minutes)', 0, 1440, 600)  # Assuming a maximum duration of 24 hours

# Preprocess user input to match the format used during model training

# Make predictions
user_input = pd.DataFrame({'Airline': [airline], 'Source': [source], 'Destination': [destination],
                           'Dep_Hour': [dep_hour], 'Dep_Min': [dep_min], 'Duration_Minutes': [duration_minutes]})
user_input_encoded = pd.get_dummies(user_input, drop_first=True)
prediction = model.predict(user_input_encoded)

# Display the result
st.write('Predicted Flight Price:', prediction)
