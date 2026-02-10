import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. LOAD MODEL
# ==========================================
# Ensure your .pkl file is in the same folder as this script
try:
    model = joblib.load("laptop_best_rf_gb_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ==========================================
# 2. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Laptop Price Predictor", layout="centered", page_icon="💻")

st.title("💻 Laptop Price Prediction")
st.info("Enter specifications below to estimate the market value in Euros (€).")

# ==========================================
# 3. USER INPUTS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Brand", 
        ['Apple', 'HP', 'Dell', 'Lenovo', 'Asus', 'Acer', 'MSI', 'Toshiba', 
         'Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 'Vero', 
         'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei'])
    
    type_name = st.selectbox("Type", 
        ['Notebook', 'Gaming', 'Ultrabook', '2 in 1 Convertible', 'Workstation', 'Netbook'])
    
    # Matching the exact categories in your training data
    opsys = st.selectbox("Operating System", 
        ['Windows 10', 'Windows 7', 'Mac OS X', 'macOS', 'Linux', 'Chrome OS', 'No OS', 'Windows 10 S'])
    
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

with col2:
    # Note: These need to match the strings in your 'Cpu_...' columns
    cpu_brand = st.selectbox("CPU Brand (Example)", 
        ['Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'Intel Celeron', 'AMD Ryzen', 'Intel Xeon'])
    
    gpu_brand = st.selectbox("GPU Brand", ['Intel', 'Nvidia', 'AMD'])
    
    inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=18.4, value=15.6, step=0.1)
    
    # We use these to create the 'ScreenResolution_...' columns
    resolution = st.selectbox("Screen Resolution", 
        ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
    
    storage = st.selectbox("Primary Storage", 
        ['128GB SSD', '256GB SSD', '512GB SSD', '1TB SSD', '500GB HDD', '1TB HDD', '128GB SSD + 1TB HDD', '256GB SSD + 1TB HDD'])

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if st.button("Calculate Estimated Price"):
    
    # A. Create Input Data
    # We map the user inputs to match the prefixes your model expects
    input_data = {
        'Inches': [inches],
        'Ram': [ram],
        'Weight': [weight],
        f'Company_{company}': [1],
        f'TypeName_{type_name}': [1],
        f'OpSys_{opsys}': [1],
        f'Cpu_{cpu_brand}': [1], # This is a simplified map; adjust based on your exact CPU strings
        f'Memory_{storage}': [1],
        f'ScreenResolution_{resolution}': [1]
    }
    
    df_input = pd.DataFrame(input_data)

    # B. Align with Model Columns (The "Reindex" trick)
    # This fills all columns the user DIDN'T select with 0
    df_final = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    # C. Predict (Model predicts on Log scale)
    prediction_log = model.predict(df_final)
    
    # D. Reverse Log Transformation (np.expm1)
    prediction_real = np.expm1(prediction_log)
    
    # E. Display Result
    st.markdown("---")
    st.subheader(f"Predicted Price: :blue[€{prediction_real[0]:,.2f}]")
    st.write("This estimate is based on current market features and Gradient Boosting analysis.")

# ==========================================
# 5. STYLING
# ==========================================
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #royalblue;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)