import streamlit as st 
import pandas as pd 
import joblib
import lightgbm as lgb 
from geopy.distance import geodesic

# Set page config for a professional look
st.set_page_config(page_title="Fraud Guard", page_icon="🛡️")

# --- DATA LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("fraud_detection_model.jb")
        encoder = joblib.load("label_encoder.jb")
        return model, encoder
    except FileNotFoundError:
        return None, None

model, encoder = load_assets()

def haversine(lat1, lon1, lat2, lon2):
    """Calculates distance between two points in km."""
    return geodesic((lat1, lon1), (lat2, lon2)).km

# --- SIDEBAR: MODEL METRICS ---
# Data hardcoded from CCFD.ipynb performance results[cite: 2]
st.sidebar.subheader("📊 Model Performance (Test Set)")
col_a, col_b = st.sidebar.columns(2)
st.sidebar.metric("ROC AUC Score", "93.7%")
col_a.metric("Recall (Fraud)", "91%")
col_b.metric("F1-Score", "0.94")
st.sidebar.divider()
st.sidebar.info("Model trained using LightGBM and SMOTE to handle class imbalance.")

# --- UI HEADER ---
st.title("🛡️ Fraud Detection System")
st.write("This application uses a LightGBM model to predict the likelihood of credit card fraud.")
st.divider()

if model is None:
    st.error("⚠️ Model files (`.jb`) not found! Please run your training script first.")
    st.stop()

# --- INPUT SECTION ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Info")
    merchant = st.text_input("Merchant Name", placeholder="e.g. Amazon")
    category = st.selectbox("Category", [
        "entertainment", "food_dining", "gas_transport", "grocery_pos", 
        "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", 
        "personal_care", "shopping_net", "shopping_pos", "travel"
    ])
    # Validation: Amount must be positive
    amt = st.number_input("Transaction Amount ($)", min_value=0.01, format="%.2f") 
    cc_num = st.number_input("Credit Card Number (Raw)", step=1, value=0)
    gender = st.selectbox("Gender", ["M", "F"])

with col2:
    st.subheader("Location & Time")
    # Validation: Latitude and Longitude ranges
    lat = st.number_input("Your Latitude", value=40.7128, min_value=-90.0, max_value=90.0, format="%.6f")
    long = st.number_input("Your Longitude", value=-74.0060, min_value=-180.0, max_value=180.0, format="%.6f")
    merch_lat = st.number_input("Merchant Latitude", value=40.7128, min_value=-90.0, max_value=90.0, format="%.6f")
    merch_long = st.number_input("Merchant Longitude", value=-74.0060, min_value=-180.0, max_value=180.0, format="%.6f")
    
    st.write("**Date/Time Details**")
    hour = st.slider("Hour (0-23)", 0, 23, 12)
    day = st.slider("Day of Month", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)

# --- PREDICTION LOGIC ---
st.divider()
if st.button("Analyze Transaction", type="primary"): 
    if merchant and category:
        # 1. Calculate Distance (Feature Engineer)[cite: 1, 2]
        distance = haversine(lat, long, merch_lat, merch_long) 

        # 2. Construct DataFrame in the EXACT order of training[cite: 1, 2]
        input_data = pd.DataFrame([[merchant, category, amt, cc_num, hour, day, month, gender, distance]],
                                  columns=['merchant', 'category', 'amt', 'cc_num', 'hour', 'day', 'month', 'gender', 'distance'])

        # 3. Apply Label Encoding with Error Handling[cite: 1]
        categorical_cols = ['merchant', 'category', 'gender']
        for col in categorical_cols: 
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except (ValueError, KeyError):
                # Inform user of unreliable prediction due to unseen data[cite: 1]
                st.warning(f"⚠️ Unseen value detected for '{col}'. Prediction may be less reliable.")
                input_data[col] = -1

        # 4. Predict[cite: 1]
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] 
        
        # 5. Display Result
        if prediction == 1:
            st.error(f"🚩 **Fraudulent Transaction Detected!**")
            st.warning(f"Risk Probability: {probability:.2%}")
        else:
            st.success(f"✅ **Legitimate Transaction**")
            st.info(f"Risk Probability: {probability:.2%}")
            
        st.write("Processed Feature Distance:", f"{distance:.2f} km")
    else:
        st.error("Please fill in the Merchant Name to continue.")