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
st.set_page_config(page_title="Agentic Supply Chain Control Tower", layout="wide")
st.title("Agentic AI Supply Chain Control Tower")
st.markdown("Predicts logistical disruptions and executes autonomous mitigation strategies.")

# Sidebar Controls
st.sidebar.header("Configure Shipment Parameters")
lead_time = st.sidebar.slider("Expected Lead Time (Days)", 1, 60, 14)
order_volume = st.sidebar.number_input("Order Volume (Units)", 100, 10000, 1500)
weather_risk = st.sidebar.slider("Weather Risk Index (0=Clear, 1=Severe)", 0.0, 1.0, 0.2)
supplier_reliability = st.sidebar.slider("Supplier Reliability Score", 0.0, 1.0, 0.85)

# Convert inputs to DataFrame
input_data = pd.DataFrame({
    'lead_time': [lead_time],
    'order_volume': [order_volume],
    'weather_index': [weather_risk],
    'supplier_reliability': [supplier_reliability]
})

st.subheader("Current Shipment Profile")
st.dataframe(input_data, use_container_width=True)

# 4. Prediction and Agentic Logic
if st.button("Analyze & Execute Decisions", type="primary"):
    # Run the Random Forest ML prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] 
    
    st.divider()
    st.subheader("Step 1: Predictive Analytics")
    
    status = "⚠️ HIGH RISK OF DISRUPTION" if prediction == 1 else "✅ SHIPMENT ON TRACK"
    st.markdown(f"**Status:** {status}")
    st.markdown(f"**Disruption Probability:** {probability:.1%}")
    
    # Visualizing Feature Importance
    importances = model.feature_importances_
    features = ['Lead Time', 'Order Volume', 'Weather Risk', 'Supplier Reliability']
    chart_data = pd.DataFrame({"Importance (%)": importances}, index=features)
    st.bar_chart(chart_data)
    
    # Find the main driver of the delay
    max_impact_feature = features[importances.argmax()]
    
    st.divider()
    st.subheader("Step 2: Agentic Decision Loop")
    
    # AGENTIC THRESHOLD: If risk is > 70%, trigger autonomous action
    if probability >= 0.70:
        st.warning("🚨 RISK EXCEEDS SAFETY THRESHOLD (70%). TRIGGERING AUTONOMOUS MITIGATION AGENT...")
        
        with st.spinner("Agent is drafting mitigation emails and contacting backup logistics partners..."):
            agent_prompt = f"""
            You are an Autonomous Supply Chain Agent.
            An active order of {order_volume} units with a {lead_time}-day window is highly likely to fail.
            The primary failure driver is the '{max_impact_feature}'.
            
            Write a formal, urgent email from 'Supply Chain Operations' to our backup vendor, 'Apex Logistics'.
            Ask if they have emergency capacity to fulfill a fallback order of {order_volume} units to bypass this disruption.
            Include specific shipment details. Keep it professional, crisp, and under 150 words. Do not include placeholders.
            """
            
            response = llm_model.generate_content(agent_prompt)
            
            # Display simulated email client
            st.info("🤖 **Agent-Generated Action: Emergency Email Drafted**")
            
            with st.container(border=True):
                st.markdown("**To:** booking@apex-logistics.com")
                st.markdown(f"**Subject:** URGENT: Fallback Fulfillment Request - {order_volume} Units")
                st.markdown("---")
                st.write(response.text)
            
            # Interactive execution button
            if st.button("Send Email & Re-route Shipment"):
                st.success("✅ Email successfully transmitted to Apex Logistics. Warehouse inventory reserve updated.")
                
    else:
        st.success("💚 Risk is within safe operating parameters. Monitoring active route conditions...")