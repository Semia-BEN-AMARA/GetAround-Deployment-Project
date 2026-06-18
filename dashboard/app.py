import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib

st.set_page_config(page_title="GetAround Dashboard", layout="wide", page_icon="🚗")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_delay_data():
    path = os.path.join(BASE_DIR, "src", "get_around_delay_analysis.xlsx")
    return pd.read_excel(path)

@st.cache_data
def load_pricing_data():
    path = os.path.join(BASE_DIR, "src", "get_around_pricing_project.csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

df_delay = load_delay_data()
df_pricing = load_pricing_data()

st.title("🚗 GetAround Business & Insights Dashboard")

# Navigation Tabs
tab1, tab2 = st.tabs(["📊 Delay Analysis (PM)", "🤖 Price Simulator (Owner)"])

# ==========================================
# TAB 1: DELAY ANALYSIS (Product Manager)
# ==========================================
with tab1:
    st.header("🔍 Delay Analysis & Late Checkouts")
    
    # Simple Metrics
    col1, col2 = st.columns(2)
    total_rentals = len(df_delay)
    late_rentals = df_delay[df_delay["delay_at_checkout_in_minutes"] > 0]
    pct_late = (len(late_rentals) / total_rentals) * 100
    
    col1.metric("Total Rentals Analysed", f"{total_rentals:,}")
    col2.metric("Late Checkout Rate", f"{pct_late:.1f} %")
    
    # Chart 1: Delay Distribution
    st.subheader("⏱️ Delay Distribution at Checkout")
    df_filtered_delays = df_delay[(df_delay["delay_at_checkout_in_minutes"] > 0) & (df_delay["delay_at_checkout_in_minutes"] <= 300)]
    fig_hist = px.histogram(
        df_filtered_delays, 
        x="delay_at_checkout_in_minutes", 
        nbins=50,
        labels={"delay_at_checkout_in_minutes": "Delay (minutes)"},
        color_discrete_sequence=["#E63946"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Chart 2: Checkin Type Impact
    st.subheader("📱 Check-in Type Impact: Connect vs Mobile")
    fig_box = px.box(
        df_delay[df_delay["delay_at_checkout_in_minutes"] > 0],
        x="checkin_type",
        y="delay_at_checkout_in_minutes",
        color="checkin_type",
        labels={"delay_at_checkout_in_minutes": "Delay (minutes)"},
        log_y=True
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# TAB 2: PRICE SIMULATOR (Owner)
# ==========================================
with tab2:
    st.header("🤖 Machine Learning Price Simulator")
    
    col_sim, col_perf = st.columns([1, 1])
    
    with col_sim:
        st.subheader("🎛️ Car Features input")
        
        model_key = st.selectbox("Car Model", sorted(df_pricing["model_key"].unique()))
        fuel = st.selectbox("Fuel Type", df_pricing["fuel"].unique())
        paint_color = st.selectbox("Color", df_pricing["paint_color"].unique())
        car_type = st.selectbox("Car Type", df_pricing["car_type"].unique())
        mileage = st.number_input("Mileage (km)", min_value=0, value=140000)
        engine_power = st.slider("Engine Power (ch)", min_value=50, max_value=500, value=120)
        
        st.markdown("**Options:**")
        private_parking = st.checkbox("Private Parking Available", value=True)
        has_gps = st.checkbox("GPS", value=True)
        has_ac = st.checkbox("Air Conditioning", value=False)
        automatic = st.checkbox("Automatic Car", value=False)
        connect = st.checkbox("GetAround Connect", value=True)
        regulator = st.checkbox("Speed Regulator", value=True)
        winter_tires = st.checkbox("Winter Tires", value=True)
        
        if st.button("Predict Optimal Price 🚀"):
            try:
                model_path = os.path.join(os.path.dirname(BASE_DIR), "api", "pricing_model.joblib")
                pipeline = joblib.load(model_path)
                
                input_data = pd.DataFrame([{
                    "model_key": model_key, "fuel": fuel, "paint_color": paint_color, "car_type": car_type,
                    "mileage": mileage, "engine_power": engine_power, "private_parking_available": private_parking,
                    "has_gps": has_gps, "has_air_conditioning": has_ac, "automatic_car": automatic,
                    "has_getaround_connect": connect, "has_speed_regulator": regulator, "winter_tires": winter_tires
                }])
                
                prediction = pipeline.predict(input_data)[0]
                st.success(f"💰 **Suggested Price: {prediction:.2f} € / day**")
            except Exception as e:
                st.error(f"Model file not found in api/ folder. Error: {e}")

    with col_perf:
        st.subheader("📈 Model Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["R² Score (Train)", "R² Score (Test)", "MAE (Mean Absolute Error)"],
            "Value": ["72.31 %", "70.58 %", "13.56 € / day"]
        })
        st.table(metrics_df)
        
        # Engine Power vs Price Scatter Plot
        fig_scatter = px.scatter(
            df_pricing.sample(1000, random_state=42),
            x="engine_power", y="rental_price_per_day",
            trendline="ols",
            title="Engine Power vs Daily Price Trend"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)