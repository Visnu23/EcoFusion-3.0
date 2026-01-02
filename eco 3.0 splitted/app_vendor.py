# ======================================================
# app_vendor.py – EcoFusion 3.0 Vendor Dashboard
# Full-featured analytics platform for vendors
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta
import sys
import os

# Add model directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------
st.set_page_config(
    page_title="🌿 EcoFusion 3.0 – Vendor Portal",
    layout="wide",
    page_icon="🔧",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# Enhanced Premium CSS
# ------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main Theme - Black & Green */
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: 900;
        margin-bottom: 0;
        background: linear-gradient(90deg, #00FF00, #00CC00, #00AA00, #00FF00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 300% 300%;
        animation: gradientFlow 4s ease infinite;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3em;
        color: #00CC00;
        margin-bottom: 1.5em;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #0A0A0A, #111111);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 8px 8px 16px #000000, -8px -8px 16px #1A1A1A;
        border: 1px solid rgba(0, 255, 0, 0.15);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00FF00, #00CC00);
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 12px 12px 24px #000000, -12px -12px 24px #222222;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: 800;
        color: #00FF00;
        margin-bottom: 5px;
    }
    
    .metric-label {
        color: #00CC00;
        font-size: 0.95em;
        font-weight: 600;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #000000 0%, #0A0A0A 100%);
        border-right: 2px solid #00AA00;
    }
    
    .sidebar-title {
        background: linear-gradient(90deg, #00AA00, #00CC00);
        color: #000000;
        font-weight: 700;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00AA00 0%, #00CC00 100%);
        color: #000000;
        border: none;
        padding: 12px 28px;
        border-radius: 25px;
        font-weight: 600;
    }
    
    h1, h2, h3 {
        color: #00FF00;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# Generate Synthetic Data
# ------------------------------------------------------
@st.cache_data
def generate_synthetic_data():
    categories = [
        "Refrigerator", "Air Conditioner", "Washing Machine", 
        "Television", "Water Heater", "LED Lighting", 
        "Computer", "Microwave", "Dishwasher", "Electric Vehicle"
    ]
    
    np.random.seed(42)
    data = []
    
    for category in categories:
        base_energy = np.random.uniform(100, 1000)
        base_co2 = base_energy * np.random.uniform(0.0004, 0.0006)
        base_cost = base_energy * np.random.uniform(0.001, 0.003)
        base_efficiency = np.random.uniform(60, 85)
        
        modern_improvement = np.random.uniform(0.7, 0.85)
        updated_improvement = np.random.uniform(0.85, 0.95)
        
        row = {
            "Category": category,
            "Old_Energy(W)": round(base_energy, 1),
            "Modern_Energy(W)": round(base_energy * modern_improvement, 1),
            "Updated_Energy(W)": round(base_energy * updated_improvement, 1),
            "Old_CO2(kg/hr)": round(base_co2, 4),
            "Modern_CO2(kg/hr)": round(base_co2 * modern_improvement, 4),
            "Updated_CO2(kg/hr)": round(base_co2 * updated_improvement, 4),
            "Old_Eff(%)": round(base_efficiency, 1),
            "Modern_Eff(%)": round(base_efficiency + np.random.uniform(5, 15), 1),
            "Updated_Eff(%)": round(base_efficiency + np.random.uniform(15, 25), 1),
            "Old_Cost($)": round(base_cost, 2),
            "Modern_Cost($)": round(base_cost * 1.2, 2),
            "Updated_Cost($)": round(base_cost * 1.5, 2),
            "Smart_Features": np.random.choice(["Basic", "Moderate", "Advanced"], p=[0.3, 0.4, 0.3]),
            "ROI_Months": np.random.randint(12, 60),
            "Maintenance_Score": np.random.randint(70, 100),
            "Warranty_Years": np.random.choice([1, 2, 3, 5, 10], p=[0.1, 0.2, 0.3, 0.3, 0.1]),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Generate time series data
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    time_series = []
    
    for category in categories:
        base_usage = np.random.uniform(50, 200)
        for date in dates:
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            noise = np.random.normal(0, 0.1)
            time_series.append({
                "Date": date,
                "Category": category,
                "Energy_Usage": round(base_usage * seasonal * (1 + noise), 2),
                "CO2_Emission": round(base_usage * seasonal * np.random.uniform(0.0004, 0.0006), 4),
                "Cost": round(base_usage * seasonal * np.random.uniform(0.001, 0.002), 2),
            })
    
    ts_df = pd.DataFrame(time_series)
    
    # Regional data
    regions = ["North", "South", "East", "West", "Central"]
    regional_data = []
    
    for region in regions:
        for category in categories:
            regional_data.append({
                "Region": region,
                "Category": category,
                "Adoption_Rate": np.random.uniform(0.3, 0.9),
                "Avg_Savings": np.random.uniform(100, 500),
                "Incentives": np.random.choice(["Yes", "No"], p=[0.6, 0.4]),
                "Grid_Stability": np.random.uniform(0.7, 0.98)
            })
    
    regional_df = pd.DataFrame(regional_data)
    
    return df, ts_df, regional_df

def analyze_efficiency(category, hours_per_day, electricity_rate, df):
    row = df[df["Category"] == category].iloc[0]
    
    energy_diff = row["Old_Energy(W)"] - row["Updated_Energy(W)"]
    co2_diff = row["Old_CO2(kg/hr)"] - row["Updated_CO2(kg/hr)"]
    eff_gain = row["Updated_Eff(%)"] - row["Old_Eff(%)"]
    
    annual_energy_saved = energy_diff * hours_per_day * 365 / 1000
    annual_co2_saved = co2_diff * hours_per_day * 365
    annual_cost_saved = annual_energy_saved * electricity_rate
    
    return {
        "energy_diff": energy_diff,
        "co2_diff": co2_diff,
        "eff_gain": eff_gain,
        "annual_energy_saved": annual_energy_saved,
        "annual_co2_saved": annual_co2_saved,
        "annual_cost_saved": annual_cost_saved,
        "row": row
    }

# ------------------------------------------------------
# Header
# ------------------------------------------------------
st.markdown("<div class='main-title'>🔧 ECOFUSION VENDOR PORTAL</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Analytics & Business Intelligence Dashboard</div>", unsafe_allow_html=True)

# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
df, ts_df, regional_df = generate_synthetic_data()

# ------------------------------------------------------
# Sidebar
# ------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>⚙️ VENDOR CONTROLS</div>", unsafe_allow_html=True)
    
    # Authentication simulation
    st.markdown("### 👤 Vendor Profile")
    vendor_name = st.text_input("Vendor ID", value="VENDOR_001", disabled=True)
    st.metric("Access Level", "Full Admin")
    
    st.markdown("---")
    
    category = st.selectbox("🏷️ Product Category", df["Category"].tolist())
    hours_per_day = st.slider("Daily Usage Hours", 1, 24, 8)
    electricity_rate = st.slider("Electricity Rate (₹/kWh)", 3, 15, 7)
    region = st.selectbox("🗺️ Target Region", regional_df["Region"].unique())
    
    st.markdown("---")
    st.markdown("### 📊 Analytics Options")
    show_raw_data = st.checkbox("Show Raw Data", False)
    show_advanced_metrics = st.checkbox("Advanced Metrics", True)
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "PDF", "JSON"])
    
    if st.button("📥 Export Full Report", use_container_width=True):
        st.success(f"Report exported as {export_format}")

# ------------------------------------------------------
# Analysis
# ------------------------------------------------------
summary = analyze_efficiency(category, hours_per_day, electricity_rate, df)
row = summary["row"]

# ------------------------------------------------------
# Metrics Dashboard
# ------------------------------------------------------
st.subheader(f"📊 {category} - VENDOR ANALYTICS")

cols = st.columns(4)
metrics = [
    (f"{summary['energy_diff']:.1f} W", "⚡ Energy Reduction"),
    (f"{summary['annual_co2_saved']:.1f} kg/yr", "🌍 CO₂ Savings"),
    (f"₹{summary['annual_cost_saved']:.0f}/yr", "💰 Cost Savings"),
    (f"+{summary['eff_gain']:.1f}%", "📈 Efficiency Gain")
]

for idx, (value, label) in enumerate(metrics):
    with cols[idx]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Business Metrics
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    st.metric("ROI Period", f"{row['ROI_Months']} months")
with col_b2:
    st.metric("Market Readiness", f"{row['Maintenance_Score']}/100")
with col_b3:
    st.metric("Warranty", f"{row['Warranty_Years']} years")
with col_b4:
    st.metric("Smart Level", row["Smart_Features"])

st.markdown("---")

# ------------------------------------------------------
# Detailed Analytics Tabs
# ------------------------------------------------------
tabs = st.tabs(["📈 ANALYTICS", "💼 BUSINESS", "🌍 REGIONAL", "📊 RAW DATA", "🔮 PREDICTIONS"])

with tabs[0]:  # Analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Energy Consumption Timeline")
        cat_ts = ts_df[ts_df["Category"] == category].copy()
        fig = px.line(cat_ts.head(90), x="Date", y="Energy_Usage",
                     title=f"{category} - 90-Day Energy Pattern",
                     color_discrete_sequence=["#00C853"])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Performance Comparison")
        comp_data = {
            "Model": ["Legacy", "Modern", "Updated"],
            "Energy": [row["Old_Energy(W)"], row["Modern_Energy(W)"], row["Updated_Energy(W)"]],
            "Efficiency": [row["Old_Eff(%)"], row["Modern_Eff(%)"], row["Updated_Eff(%)"]]
        }
        fig2 = px.bar(comp_data, x="Model", y=["Energy", "Efficiency"], barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

with tabs[1]:  # Business
    st.markdown("### 💰 Financial Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Investment Analysis**
        - Initial Cost: ₹{row['Updated_Cost($)'] * 100:,.0f}
        - Annual Return: ₹{summary['annual_cost_saved']:,.0f}
        - Payback: {row['ROI_Months']} months
        - NPV (5yr): ₹{summary['annual_cost_saved'] * 5 - row['Updated_Cost($)'] * 100:,.0f}
        """)
    
    with col2:
        st.success(f"""
        **Market Position**
        - Smart Features: {row['Smart_Features']}
        - Warranty: {row['Warranty_Years']} years
        - Maintenance Score: {row['Maintenance_Score']}/100
        - Competitive Edge: High
        """)
    
    with col3:
        st.warning(f"""
        **Sales Projections**
        - Target Market: {region}
        - Expected Adoption: 65%
        - Revenue Potential: High
        - Competition: Moderate
        """)

with tabs[2]:  # Regional
    st.markdown("### 🗺️ Regional Market Analysis")
    
    regional_filtered = regional_df[regional_df["Category"] == category]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(regional_filtered, x="Region", y="Adoption_Rate",
                    color="Adoption_Rate", color_continuous_scale=["#AEEA00", "#00C853"])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(regional_filtered, x="Adoption_Rate", y="Avg_Savings",
                        size="Grid_Stability", color="Region")
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:  # Raw Data
    st.markdown("### 📊 Raw Data Tables")
    
    st.markdown("#### Product Specifications")
    st.dataframe(df, use_container_width=True)
    
    st.markdown("#### Time Series Data (Last 30 Days)")
    st.dataframe(ts_df[ts_df["Category"] == category].tail(30), use_container_width=True)
    
    st.markdown("#### Regional Statistics")
    st.dataframe(regional_df[regional_df["Category"] == category], use_container_width=True)

with tabs[4]:  # Predictions
    st.markdown("### 🔮 AI-Powered Market Predictions")
    
    years = np.arange(2024, 2034)
    future_adoption = [0.3 + (i * 0.06) for i in range(len(years))]
    future_revenue = [summary['annual_cost_saved'] * 1000 * (1.15 ** i) for i in range(len(years))]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Scatter(x=years, y=future_adoption, 
                                   line=dict(color="#00C853", width=3)))
        fig.update_layout(title="Adoption Rate Forecast", yaxis_title="Adoption %")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Scatter(x=years, y=future_revenue,
                                   line=dict(color="#64DD17", width=3)))
        fig.update_layout(title="Revenue Projection", yaxis_title="Revenue (₹)")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
# Footer
# ------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <div style='color: #00C853; font-weight: 700; font-size: 1.1em;'>🔧 ECOFUSION VENDOR PORTAL</div>
    <div>© {datetime.now().year} EcoFusion 3.0 | Vendor Access Level: Full Admin</div>
    <div style='font-size: 0.85em; margin-top: 10px;'>
        Session: {vendor_name} | Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</div>
""", unsafe_allow_html=True)