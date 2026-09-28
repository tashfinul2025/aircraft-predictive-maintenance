import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Military Aircraft Health Predictor", layout="centered")
st.title("🛩️ Aircraft Engine Health & RUL Predictor")
st.write("Adjust the live engine sensor readings below to predict Remaining Useful Life.")

# Load your trained model brain safely
try:
    with open('engine_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Could not find 'engine_model.pkl'. Please make sure it's in the same directory.")

st.subheader("📊 Live Sensor Inputs")

# Create interactive sliders for the key engine sensors
sensor_2 = st.slider("Sensor 2 (Core Temperature)", min_value=640.0, max_value=645.0, value=642.5, step=0.1)
sensor_3 = st.slider("Sensor 3 (Bypass Ratio)", min_value=1580.0, max_value=1600.0, value=1590.0, step=0.5)
sensor_4 = st.slider("Sensor 4 (Total Pump Pressure)", min_value=1390.0, max_value=1420.0, value=1405.0, step=0.5)

# Build a matching array of 21 inputs for our model (since we have 21 active columns)
# We fill it with generic placeholders, then overwrite the specific sensor indexes
inputs = np.zeros(21)
inputs[0] = sensor_2  # Map to Sensor 2 slot
inputs[1] = sensor_3  # Map to Sensor 3 slot
inputs[2] = sensor_4  # Map to Sensor 4 slot

# Trigger the prediction calculation when the user interacts
if st.button("🔮 Calculate Remaining Useful Life"):
    prediction = model.predict(inputs.reshape(1, -1))[0]
    
    if prediction < 0:
        st.error("🚨 CRITICAL FAILURE IMMINENT: Maintenance Overdue!")
    else:
        st.metric(label="Predicted Flight Cycles Left", value=f"{int(prediction)} cycles")
