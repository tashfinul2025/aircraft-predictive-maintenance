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

# 3. Interactive Web Sliders (Ranges chosen based on typical FD001 engine values)
sensor_2 = st.slider("Sensor 2 (Core Temperature)", min_value=640.0, max_value=645.0, value=642.5, step=0.1)
sensor_3 = st.slider("Sensor 3 (Bypass Ratio)", min_value=1580.0, max_value=1610.0, value=1590.0, step=0.5)
sensor_4 = st.slider("Sensor 4 (Total Pump Pressure)", min_value=1390.0, max_value=1430.0, value=1405.0, step=0.5)

# 4. Construct the Full 14-Feature Input Array
# Because we dropped 7 dead columns earlier (leaving 14 features out of the 21 sensors/settings), 
# we construct an array of the 14 remaining active variables.
# We populate it with baseline averages so the linear math equation doesn't explode.
inputs = np.array([
    sensor_2,   # Sensor 2 (Active Slider)
    sensor_3,   # Sensor 3 (Active Slider)
    sensor_4,   # Sensor 4 (Active Slider)
    553.5,      # Sensor 7 Average
    2388.0,     # Sensor 8 Average
    9050.0,     # Sensor 9 Average
    47.5,       # Sensor 11 Average
    521.8,      # Sensor 12 Average
    2388.0,     # Sensor 13 Average
    8135.0,     # Sensor 14 Average
    8.4,        # Sensor 15 Average
    392.0,      # Sensor 17 Average
    39.0,       # Sensor 20 Average
    23.3        # Sensor 21 Average
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

