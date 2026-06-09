import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('house_price_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 California House Price Predictor")
st.write("Built by **Priyanshu Mewa** | B.Tech ECE, IIIT Una")
st.markdown("---")

st.subheader("Enter house details:")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.5)
    HouseAge = st.slider("House Age (years)", 1, 52, 20)
    AveRooms = st.slider("Average Rooms", 1.0, 20.0, 5.0)
    AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)

with col2:
    Population = st.slider("Population", 3, 10000, 1200)
    AveOccup = st.slider("Average Occupants", 1.0, 10.0, 3.0)
    Latitude = st.slider("Latitude", 32.5, 42.0, 35.0)
    Longitude = st.slider("Longitude", -124.0, -114.0, -119.0)

if st.button("🔍 Predict Price"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(features)[0]
    price_usd = prediction * 100000

    st.success(f"🏠 Predicted Price: **${price_usd:,.0f}**")
    st.caption(f"Raw model output: {prediction:.4f} (×$100,000)")

st.markdown("---")
st.markdown("### 📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", "0.7743")
col2.metric("RMSE", "0.4652")
col3.metric("MAE", "0.3110")

st.caption("Model: Random Forest Regressor | Dataset: California Housing (20,640 samples)")