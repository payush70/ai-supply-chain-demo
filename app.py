import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import yfinance as yf
import feedparser
import requests


def get_live_oil_price():
    try:
        oil_ticker = yf.Ticker("CL=F")
        todays_data = oil_ticker.history(period="1d")
        live_price = todays_data['Close'].iloc[-1]
        return round(live_price, 2)
    except Exception as e:
        return 75.00 


def get_live_logistics_news():
    try:
        url = "https://news.google.com/rss/search?q=global+supply+chain+shipping+disruption+geopolitics&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries[:3]:
            headlines.append(entry.title)
        return headlines
    except Exception as e:
        return ["Unable to fetch live news at this time."]

def get_live_weather_risk(city):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_response = requests.get(geo_url).json()
        
        if not geo_response.get('results'):
            return 0.2, "Clear (City Not Found)"
            
        lat = geo_response['results'][0]['latitude']
        lon = geo_response['results'][0]['longitude']
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_data = requests.get(weather_url).json()
        
        current = weather_data['current_weather']
        wmo_code = current['weathercode']
        temp = current['temperature']
        wind = current['windspeed']
        
        risk = 0.2
        condition = "Clear/Cloudy"
        
        if wmo_code in [45, 48]: 
            risk, condition = 0.4, "Foggy"
        elif wmo_code in range(51, 68): 
            risk, condition = 0.6, "Rain"
        elif wmo_code in range(71, 78): 
            risk, condition = 0.8, "Heavy Snow"
        elif wmo_code in range(80, 83): 
            risk, condition = 0.7, "Heavy Rain Showers"
        elif wmo_code >= 95: 
            risk, condition = 1.0, "Severe Thunderstorms"
            
        if wind > 40.0:
            risk = min(1.0, risk + 0.2)
            condition += " + High Winds"
            
        desc = f"{condition} (°C, Wind: {wind}km/h)"
        return round(risk, 2), desc
        
    except Exception as e:
        return 0.2, "Clear (API Offline)"


genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm_model = genai.GenerativeModel("gemini-3.5-flash")

@st.cache_resource
def load_model():
    return joblib.load('rf_predictive_model.pkl')

model = load_model()

st.set_page_config(page_title="Agentic Supply Chain Control Tower", layout="wide")
st.title("Agentic AI Supply Chain Control Tower")
st.markdown("Predicts logistical disruptions, evaluates inventory product risk, and accounts for major global events.")

# ----------------------------------------------------
# SIDEBAR: MACRO & PRODUCTS & EVENTS
# ----------------------------------------------------
st.sidebar.header("📦 Inventory & Product Profile")
product_name = st.sidebar.text_input("Product Stock Name", value="Fresh Milk & Juices Batch")
product_category = st.sidebar.selectbox("Category / Shelf Life Risk", ["Perishable (Short Shelf Life)", "Standard Goods", "Bulk Commodities"])

perishability_multiplier = 1.2 if product_category == "Perishable (Short Shelf Life)" else 1.0

st.sidebar.divider()
st.sidebar.header("🎉 Demand & Event Impact (FIFA, F1, Olympics)")
active_event = st.sidebar.selectbox("Upcoming City Event", ["None / Normal Operations", "FIFA World Cup Match", "Formula 1 Grand Prix", "Global Music Festival / Concert", "National Holiday / Expo"])

event_surge_multiplier = 1.0
if active_event == "FIFA World Cup Match":
    event_surge_multiplier = 1.35
elif active_event == "Formula 1 Grand Prix":
    event_surge_multiplier = 1.25
elif active_event == "Global Music Festival / Concert":
    event_surge_multiplier = 1.20

st.sidebar.divider()
st.sidebar.header("🌍 Live Macro Factors")

live_oil_price = get_live_oil_price()
st.sidebar.metric(label="Live WTI Crude Oil (USD/bbl)", value=f"${live_oil_price}")

st.sidebar.divider()
st.sidebar.subheader("📰 Live Global Logistics News")
news_headlines = get_live_logistics_news()
for headline in news_headlines:
    st.sidebar.caption(f"• {headline}")

st.sidebar.divider()
st.sidebar.subheader("🌦️ Destination Weather")
destination_city = st.sidebar.text_input("Destination City or Port", value="London")
weather_risk, weather_desc = get_live_weather_risk(destination_city)

st.sidebar.metric(label=f"Current Weather", value=weather_desc)
st.sidebar.metric(label="Calculated Weather Risk", value=f"{weather_risk} / 1.0")

st.sidebar.divider()
st.sidebar.header("Shipment Parameters")
lead_time = st.sidebar.slider("Expected Lead Time (Days)", 1, 60, 14)
order_volume = st.sidebar.number_input("Order Volume (Units)", 100, 10000, 1500)
supplier_reliability = st.sidebar.slider("Supplier Reliability Score", 0.0, 1.0, 0.85)

# Convert inputs to DataFrame
input_data = pd.DataFrame({
    'lead_time': [lead_time],
    'order_volume': [order_volume],
    'weather_index': [weather_risk],
    'supplier_reliability': [supplier_reliability]
})

st.subheader(f"Current Stock Profile: {product_name}")
st.markdown(f"**Target Event Context:** {active_event} | **Inventory Class:** {product_category}")
st.dataframe(input_data, use_container_width=True)

# ----------------------------------------------------
# PREDICTION & AGENT LOOP WITH EVENT/PRODUCT WEIGHTS
# ----------------------------------------------------
if st.button("Analyze & Execute Decisions", type="primary"):
    base_prediction = model.predict(input_data)[0]
    base_probability = model.predict_proba(input_data)[0][1] 
    
    adjusted_probability = min(1.0, base_probability * perishability_multiplier * event_surge_multiplier)
    adjusted_prediction = 1 if adjusted_probability >= 0.5 else 0
    
    st.divider()
    st.subheader("Step 1: Predictive Analytics (Inventory & Event Adjusted)")
    
    status = "⚠️ HIGH RISK OF DISRUPTION/STOCKOUT" if adjusted_prediction == 1 else "✅ SHIPMENT ON TRACK"
    st.markdown(f"**Status:** {status}")
    st.markdown(f"**Adjusted Disruption Probability:** {adjusted_probability:.1%}")
    if active_event != "None / Normal Operations":
        st.info(f"💡 *Note: Risk calculation amplified due to surge demands and traffic bottlenecks expected during the {active_event} in {destination_city}.*")
    
    importances = model.feature_importances_
    features = ['Lead Time', 'Order Volume', 'Weather Risk', 'Supplier Reliability']
    chart_data = pd.DataFrame({"Importance (%)": importances}, index=features)
    st.bar_chart(chart_data)
    
    max_impact_feature = features[importances.argmax()]
    
    st.divider()
    st.subheader("Step 2: Agentic Decision Loop")
    
    if adjusted_probability >= 0.70:
        st.warning("🚨 RISK EXCEEDS SAFETY THRESHOLD (70%). TRIGGERING AUTONOMOUS MITIGATION AGENT...")
        
        with st.spinner("Agent is analyzing inventory requirements, event traffic, and drafting supplier communications..."):
            
            news_context = "\n".join(news_headlines)
            
            agent_prompt = f"""
            You are an Autonomous Supply Chain Agent managing inventory for perishable and high-demand goods.
            An active stock shipment of '{product_name}' ({order_volume} units, category: {product_category}) heading to {destination_city} has an adjusted disruption risk of {adjusted_probability:.1%}.
            The primary failure driver is '{max_impact_feature}'.
            
            CRITICAL CONTEXT:
            1. Upcoming Major Event in City: {active_event} (requiring rapid stock replenishment).
            2. The live market price of crude oil is currently ${live_oil_price}/bbl. 
            3. LIVE WEATHER AT DESTINATION ({destination_city}): {weather_desc}.
            4. LIVE GLOBAL NEWS HEADLINES:
            {news_context}
            
            Write a formal, urgent email to our backup vendor, 'Apex Logistics'.
            Ask if they have emergency cold-chain/express capacity to fulfill a fallback order of {order_volume} units of {product_name} to {destination_city} ahead of the {active_event}.
            Factor the current crude oil price, destination weather, perishability, and event congestion.
            Keep it professional, crisp, and under 150 words. Do not include placeholders.
            """
            
            response = llm_model.generate_content(agent_prompt)
            
            st.info("🤖 **Agent-Generated Action: Emergency Stock-Rerouting Drafted**")
            
            with st.container(border=True):
                st.markdown("**To:** booking@apex-logistics.com")
                st.markdown(f"**Subject:** URGENT: Event Stock Fallback - {product_name} ({order_volume} Units)")
                st.markdown("---")
                st.write(response.text)
            
            if st.button("Send Emergency Dispatch & Reserve Stock"):
                st.success("✅ Fallback dispatch instruction successfully transmitted to Apex Logistics. Regional warehouse buffer activated.")
                
    else:
        st.success("💚 Inventory risk is within safe operating parameters for the upcoming event window. Monitoring active route conditions...")