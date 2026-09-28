import streamlit as st
import pickle
import numpy as np

# 1. Page Styling and Header
st.set_page_config(page_title="Military Aircraft Health Predictor", layout="centered")
st.title("🛩️ Aircraft Engine Health & RUL Predictor")
st.write("Adjust the live engine sensor readings below to predict Remaining Useful Life.")

# 2. Load the trained Scikit-Learn Model
try:
    with open('engine_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Could not find 'engine_model.pkl'. Please make sure it's in the same directory.")

st.subheader("📊 Live Sensor Inputs")

# 3. Interactive Web Sliders
sensor_2 = st.slider("Sensor 2 (Core Temperature)", min_value=640.0, max_value=645.0, value=642.5, step=0.1)
sensor_3 = st.slider("Sensor 3 (Bypass Ratio)", min_value=1580.0, max_value=1610.0, value=1590.0, step=0.5)
sensor_4 = st.slider("Sensor 4 (Total Pump Pressure)", min_value=1390.0, max_value=1430.0, value=1405.0, step=0.5)

# 4. Construct the Full 21-Feature Input Array to match your trained model
# This builds a full baseline array of typical averages for all 21 active sensor/setting slots
inputs = np.array([
    0.02, 0.003, 100.0,     # Settings 1, 2, 3
    642.5,                  # Sensor 1
    sensor_2,               # Sensor 2 (Active Slider)
    sensor_3,               # Sensor 3 (Active Slider)
    sensor_4,               # Sensor 4 (Active Slider)
    553.5, 2388.0, 9050.0,  # Sensors 5, 6, 7
    1.3, 47.5, 521.8,       # Sensors 8, 9, 10
    2388.0, 8135.0, 8.4,    # Sensors 11, 12, 13
    0.03, 392.0, 2388.0,    # Sensors 14, 15, 16
    100.0, 39.0, 23.3       # Sensors 17, 18, 19
])

# 5. Prediction Execution
if st.button("🔮 Calculate Remaining Useful Life"):
    # Reshape the 1D array into a 2D row format required by Scikit-Learn
    prediction = model.predict(inputs.reshape(1, -1))
    predicted_value = int(prediction[0])
    
    # Render Output Metrics based on threshold conditions
    if predicted_value <= 0:
        st.error("🚨 CRITICAL FAILURE IMMINENT: Pull Engine into Hangar Immediately!")
    elif predicted_value < 30:
        st.warning(f"⚠️ SCHEDULE MAINTENANCE SOON: Only {predicted_value} flight cycles remaining.")
    else:
        st.success(f"✅ ENGINE HEALTHY: Predicted Remaining Useful Life is {predicted_value} flight cycles.")
