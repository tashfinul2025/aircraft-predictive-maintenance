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

# 4. Check the Model's Expected Input Size Dynamically
# Scikit-learn models have a hidden property showing how many features they expect
try:
    expected_features = model.n_features_in_
except AttributeError:
    expected_features = 24  # Standard default if not exposed

# 5. Build a dynamic array that scales perfectly to match your specific model
# We fill the first 3 slots with your live sliders, and pad the rest with safe averages
base_values = [sensor_2, sensor_3, sensor_4]
default_padding = [553.5, 2388.0, 9050.0, 1.3, 47.5, 521.8, 2388.0, 8135.0, 8.4, 0.03, 392.0, 2388.0, 100.0, 39.0, 23.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Combine your sliders with just enough padding to equal the model's exact shape
needed_padding = expected_features - len(base_values)
inputs = np.array(base_values + default_padding[:needed_padding])

# 6. Prediction Execution
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
