import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai

# 1. Configure Gemini API 
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm_model = genai.GenerativeModel("gemini-3.5-flash")

# 2. Load the Random Forest Model
@st.cache_resource
def load_model():
    return joblib.load('rf_predictive_model.pkl')

model = load_model()

# 3. UI Setup
st.set_page_config(page_title="Supply Chain Predictor", layout="wide")
st.title("AI Supply Chain Disruption Predictor")

st.sidebar.header("Configure Shipment Parameters")
lead_time = st.sidebar.slider("Expected Lead Time (Days)", 1, 60, 14)
order_volume = st.sidebar.number_input("Order Volume (Units)", 100, 10000, 1500)
weather_risk = st.sidebar.slider("Weather Risk Index (0=Clear, 1=Severe)", 0.0, 1.0, 0.2)
supplier_reliability = st.sidebar.slider("Supplier Reliability Score", 0.0, 1.0, 0.85)

input_data = pd.DataFrame({
    'lead_time': [lead_time],
    'order_volume': [order_volume],
    'weather_index': [weather_risk],
    'supplier_reliability': [supplier_reliability]
})

st.subheader("Current Shipment Profile")
st.dataframe(input_data, use_container_width=True)

# 4. Prediction and GenAI Explanation
if st.button("Predict Disruption Risk", type="primary"):
    # Machine Learning Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] 
    
    st.divider()
    st.subheader("Prediction Results")
    
    status = "⚠️ HIGH RISK OF DISRUPTION" if prediction == 1 else "✅ SHIPMENT ON TRACK"
    st.markdown(f"**Status:** {status}")
    st.markdown(f"**Disruption Probability:** {probability:.1%}")
    
    # Generative AI Action Plan (Jarvis)
    with st.spinner("Jarvis is analyzing the logistics data..."):
        prompt = f"""
        You are Jarvis, an expert AI logistics assistant. 
        A shipment has the following parameters: Lead Time: {lead_time} days, Volume: {order_volume}, 
        Weather Risk: {weather_risk}, Supplier Reliability: {supplier_reliability}.
        Our predictive model indicates a disruption probability of {probability:.1%}.
        Write a concise, 3-sentence action plan for the supply chain manager.
        """
        response = llm_model.generate_content(prompt)
        
        st.info("🤖 **Jarvis's Action Plan:**")
        st.write(response.text)