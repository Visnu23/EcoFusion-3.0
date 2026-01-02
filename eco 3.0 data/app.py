# ======================================================
# app.py – EcoFusion 3.0 Combined Dashboard
# Single app with Vendor and Customer views
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import sys
import os

# Add model directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------
# Page Config
# ------------------------------------------------------
st.set_page_config(
    page_title="🌿 EcoFusion 3.0 - Smart Energy Platform",
    layout="wide",
    page_icon="🌿",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# CSS Styling
# ------------------------------------------------------
st.markdown(
    """
    <style>
    /* Common Styles */
    .main-title {
        text-align: center;
        font-size: 2.8em;
        font-weight: 800;
        margin-bottom: 5px;
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
        font-size: 1.2em;
        color: #00CC00;
        margin-bottom: 1.5em;
    }
    
    /* Mode Selector */
    .mode-selector {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,200,83,0.3);
    }
    
    .mode-button {
        display: inline-block;
        background: linear-gradient(135deg, #00AA00, #00CC00);
        color: white;
        padding: 20px 40px;
        margin: 10px;
        border-radius: 15px;
        font-weight: 700;
        font-size: 1.2em;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    /* Vendor Styles */
    .vendor-metric-card {
        background: linear-gradient(145deg, #0A0A0A, #111111);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 8px 8px 16px #000000, -8px -8px 16px #1A1A1A;
        border: 1px solid rgba(0, 255, 0, 0.15);
        transition: all 0.4s ease;
    }
    
    .vendor-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00FF00, #00CC00);
    }
    
    .vendor-metric-card:hover {
        transform: translateY(-5px);
    }
    
    .vendor-metric-value {
        font-size: 2em;
        font-weight: 800;
        color: #00FF00;
        margin-bottom: 5px;
    }
    
    .vendor-metric-label {
        color: #00CC00;
        font-size: 0.95em;
        font-weight: 600;
    }
    
    /* Customer Styles */
    .customer-big-metric {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,200,83,0.2);
    }
    
    .customer-metric-value {
        font-size: 3em;
        font-weight: 800;
        color: #00C853;
        margin-bottom: 10px;
    }
    
    .customer-metric-label {
        font-size: 1.2em;
        color: #388E3C;
        font-weight: 600;
    }
    
    .benefit-item {
        background: #E8F5E9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00C853;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #FFF9C4, #FFF59D);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border-left: 5px solid #FBC02D;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00AA00, #00CC00);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00CC00, #00FF00);
        transform: translateY(-2px);
    }
    
    /* Sidebar */
    .stSidebar {
        background: linear-gradient(180deg, #000000 0%, #0A0A0A 100%);
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
    
    /* Headers */
    h1, h2, h3 {
        color: #00C853;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# Data Generation Functions
# ------------------------------------------------------
@st.cache_data
def generate_vendor_data():
    """Generate comprehensive data for vendor view"""
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
    
    # Time series data
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

@st.cache_data
def generate_customer_data():
    """Generate simplified data for customer view"""
    appliances = {
        "Refrigerator": {"energy": 150, "icon": "❄️", "usage": "24/7"},
        "Air Conditioner": {"energy": 1500, "icon": "🌡️", "usage": "8 hrs/day"},
        "Washing Machine": {"energy": 500, "icon": "🧺", "usage": "2 hrs/day"},
        "Television": {"energy": 100, "icon": "📺", "usage": "6 hrs/day"},
        "Water Heater": {"energy": 2000, "icon": "🚿", "usage": "2 hrs/day"},
        "LED Lighting": {"energy": 60, "icon": "💡", "usage": "8 hrs/day"},
        "Computer": {"energy": 200, "icon": "💻", "usage": "8 hrs/day"},
        "Microwave": {"energy": 1200, "icon": "🍽️", "usage": "1 hr/day"},
        "Dishwasher": {"energy": 1800, "icon": "🍴", "usage": "1 hr/day"},
        "Electric Vehicle": {"energy": 7000, "icon": "🚗", "usage": "2 hrs/day"}
    }
    
    savings_data = {}
    for name, info in appliances.items():
        old_energy = info["energy"]
        new_energy = old_energy * 0.6  # 40% savings
        savings_data[name] = {
            "icon": info["icon"],
            "old_energy": old_energy,
            "new_energy": new_energy,
            "savings_percent": 40,
            "usage": info["usage"],
            "monthly_savings": (old_energy - new_energy) * 0.007 * 30
        }
    
    return savings_data

def analyze_vendor_efficiency(category, hours_per_day, electricity_rate, df):
    """Calculate vendor analytics"""
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

def calculate_customer_savings(appliance, hours, rate, data):
    """Calculate customer savings"""
    info = data[appliance]
    old_watts = info["old_energy"]
    new_watts = info["new_energy"]
    
    daily_old = (old_watts * hours) / 1000
    daily_new = (new_watts * hours) / 1000
    
    monthly_old = daily_old * 30 * rate
    monthly_new = daily_new * 30 * rate
    monthly_savings = monthly_old - monthly_new
    
    yearly_savings = monthly_savings * 12
    co2_saved = (daily_old - daily_new) * 30 * 0.5
    
    return {
        "monthly_old": monthly_old,
        "monthly_new": monthly_new,
        "monthly_savings": monthly_savings,
        "yearly_savings": yearly_savings,
        "co2_saved": co2_saved,
        "energy_reduction": ((old_watts - new_watts) / old_watts) * 100
    }

# ------------------------------------------------------
# Initialize Session State
# ------------------------------------------------------
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = None

# ------------------------------------------------------
# Header
# ------------------------------------------------------
st.markdown("<div class='main-title'>🌿 ECOFUSION 3.0</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Energy Intelligence Platform</div>", unsafe_allow_html=True)

# ------------------------------------------------------
# Mode Selection (if not selected)
# ------------------------------------------------------
if st.session_state.view_mode is None:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class='mode-selector'>
            <h2 style='color: #00C853; margin-bottom: 20px;'>👋 Welcome! Choose Your View</h2>
            <p style='font-size: 1.1em; color: #666; margin-bottom: 30px;'>
                Select the interface that best suits your needs
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #E8F5E9, #C8E6C9); 
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 12px rgba(0,200,83,0.2);'>
                <div style='font-size: 3em; margin-bottom: 15px;'>🌱</div>
                <h3 style='color: #00C853;'>Customer View</h3>
                <p style='color: #666; margin: 15px 0;'>
                    Simple, user-friendly interface to calculate your personal savings
                </p>
                <ul style='text-align: left; color: #666; list-style: none; padding: 0;'>
                    <li>✓ Easy savings calculator</li>
                    <li>✓ Clear visualizations</li>
                    <li>✓ Personal benefits</li>
                    <li>✓ Environmental impact</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🌱 I'm a Customer", use_container_width=True, key="customer_btn"):
                st.session_state.view_mode = "customer"
                st.rerun()
        
        with col_b:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1A1A1A, #0A0A0A); 
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 12px rgba(0,255,0,0.2);'>
                <div style='font-size: 3em; margin-bottom: 15px;'>🔧</div>
                <h3 style='color: #00FF00;'>Vendor View</h3>
                <p style='color: #00CC00; margin: 15px 0;'>
                    Advanced analytics and business intelligence dashboard
                </p>
                <ul style='text-align: left; color: #00CC00; list-style: none; padding: 0;'>
                    <li>✓ Full data analytics</li>
                    <li>✓ Market insights</li>
                    <li>✓ Business metrics</li>
                    <li>✓ Export capabilities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔧 I'm a Vendor", use_container_width=True, key="vendor_btn"):
                st.session_state.view_mode = "vendor"
                st.rerun()

# ------------------------------------------------------
# CUSTOMER VIEW
# ------------------------------------------------------
elif st.session_state.view_mode == "customer":
    
    # Switch mode button in sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>🌱 CUSTOMER PORTAL</div>", unsafe_allow_html=True)
        
        if st.button("🔄 Switch to Vendor View", use_container_width=True):
            st.session_state.view_mode = "vendor"
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🏠 Tell Us About Your Appliance")
        
        customer_data = generate_customer_data()
        
        appliance_options = [f"{info['icon']} {name}" for name, info in customer_data.items()]
        selected = st.selectbox("What appliance do you want to upgrade?", appliance_options)
        appliance = selected.split(" ", 1)[1]
        
        st.markdown("---")
        st.markdown("### ⚡ Your Usage")
        
        hours = st.slider("How many hours per day do you use it?", 1, 24, 8)
        rate = st.slider("What's your electricity rate? (₹/kWh)", 3, 15, 7)
        
        st.markdown("---")
        st.info(f"""
        **Quick Info:**
        
        {customer_data[appliance]['icon']} Typical usage: {customer_data[appliance]['usage']}
        
        💡 Your input: {hours} hours/day
        
        ⚡ Rate: ₹{rate}/kWh
        """)
    
    # Calculate savings
    savings = calculate_customer_savings(appliance, hours, rate, customer_data)
    
    # Hero metrics
    st.markdown("## 💰 Your Potential Savings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='customer-big-metric'>
            <div class='customer-metric-value'>₹{savings['monthly_savings']:.0f}</div>
            <div class='customer-metric-label'>Per Month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='customer-big-metric'>
            <div class='customer-metric-value'>₹{savings['yearly_savings']:.0f}</div>
            <div class='customer-metric-label'>Per Year</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='customer-big-metric'>
            <div class='customer-metric-value'>{savings['co2_saved']:.0f} kg</div>
            <div class='customer-metric-label'>CO₂ Saved Monthly</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visual comparison
    st.markdown("---")
    st.markdown("## 📊 Easy Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💡 Your Current Bill vs New Bill")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Current Bill", "New Bill"],
            y=[savings['monthly_old'], savings['monthly_new']],
            marker_color=["#EF5350", "#00C853"],
            text=[f"₹{savings['monthly_old']:.0f}", f"₹{savings['monthly_new']:.0f}"],
            textposition='auto',
        ))
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="Monthly Cost (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🌍 Your Environmental Impact")
        
        fig = go.Figure(data=[go.Pie(
            labels=["Energy Saved", "Energy Used"],
            values=[savings['energy_reduction'], 100 - savings['energy_reduction']],
            marker_colors=["#00C853", "#E0E0E0"],
            hole=0.4
        )])
        fig.update_layout(
            height=400,
            showlegend=True,
            annotations=[dict(text=f"{savings['energy_reduction']:.0f}%<br>Saved", 
                             x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Benefits
    st.markdown("---")
    st.markdown("## ✨ What This Means For You")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trees = savings['co2_saved'] / 21
        
        st.markdown(f"""
        <div class='benefit-item'>
            <h3>🌳 Environmental Impact</h3>
            <p><strong>You'll save {savings['co2_saved']:.0f} kg of CO₂ every month!</strong></p>
            <p>That's like planting <strong>{trees:.1f} trees</strong> every month</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='benefit-item'>
            <h3>💰 Money Saved</h3>
            <p><strong>Save ₹{savings['monthly_savings']:.0f} every month</strong></p>
            <p>In 5 years, you'll save: <strong>₹{savings['yearly_savings'] * 5:,.0f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='benefit-item'>
            <h3>⚡ Energy Reduction</h3>
            <p><strong>{savings['energy_reduction']:.0f}% less energy used</strong></p>
            <p>Your new {appliance} will be much more efficient!</p>
        </div>
        """, unsafe_allow_html=True)
        
        payback = (customer_data[appliance]['new_energy'] * 100) / savings['monthly_savings'] if savings['monthly_savings'] > 0 else 0
        
        st.markdown(f"""
        <div class='benefit-item'>
            <h3>📅 Payback Time</h3>
            <p><strong>Your investment pays back in {payback:.0f} months</strong></p>
            <p>After that, it's pure savings!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Timeline
    st.markdown("---")
    st.markdown("## 📈 Your Savings Over Time")
    
    months = list(range(1, 61))
    cumulative = [savings['monthly_savings'] * m for m in months]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=cumulative,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#00C853', width=3),
        fillcolor='rgba(0, 200, 83, 0.2)'
    ))
    
    fig.update_layout(
        title="Your Total Savings Over 5 Years",
        xaxis_title="Months",
        yaxis_title="Total Savings (₹)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # CTA
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🌟 Get Your Personalized Upgrade Plan", use_container_width=True):
            st.balloons()
            st.success("🎉 Great choice! We'll send you a detailed report soon!")

# ------------------------------------------------------
# VENDOR VIEW
# ------------------------------------------------------
elif st.session_state.view_mode == "vendor":
    
    # Load vendor data
    df, ts_df, regional_df = generate_vendor_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>🔧 VENDOR PORTAL</div>", unsafe_allow_html=True)
        
        if st.button("🔄 Switch to Customer View", use_container_width=True):
            st.session_state.view_mode = "customer"
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 👤 Vendor Profile")
        st.text_input("Vendor ID", value="VENDOR_001", disabled=True)
        st.metric("Access Level", "Full Admin")
        
        st.markdown("---")
        
        category = st.selectbox("🏷️ Product Category", df["Category"].tolist())
        hours_per_day = st.slider("Daily Usage Hours", 1, 24, 8)
        electricity_rate = st.slider("Electricity Rate (₹/kWh)", 3, 15, 7)
        region = st.selectbox("🗺️ Target Region", regional_df["Region"].unique())
        
        st.markdown("---")
        st.markdown("### 📊 Analytics Options")
        show_raw_data = st.checkbox("Show Raw Data", False)
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "PDF", "JSON"])
        
        if st.button("📥 Export Report", use_container_width=True):
            st.success(f"Report exported as {export_format}")
    
    # Analysis
    summary = analyze_vendor_efficiency(category, hours_per_day, electricity_rate, df)
    row = summary["row"]
    
    # Metrics
    st.markdown(f"## 📊 {category} - VENDOR ANALYTICS")
    
    cols = st.columns(4)
    metrics = [
        (f"{summary['energy_diff']:.1f} W", "⚡ Energy Reduction"),
        (f"{summary['annual_co2_saved']:.1f} kg/yr", "🌍 CO₂ Savings"),
        (f"₹{summary['annual_cost_saved']:.0f}/yr", "💰 Cost Savings"),
        (f"+{summary['eff_gain']:.1f}%", "📈 Efficiency Gain")
    ]
    
    for idx, (value, label) in enumerate(metrics):
        with cols[idx]:
            st.markdown(f"""
            <div class='vendor-metric-card'>
                <div class='vendor-metric-value'>{value}</div>
                <div class='vendor-metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Business metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROI Period", f"{row['ROI_Months']} months")
    with col2:
        st.metric("Market Readiness", f"{row['Maintenance_Score']}/100")
    with col3:
        st.metric("Warranty", f"{row['Warranty_Years']} years")
    with col4:
        st.metric("Smart Level", row["Smart_Features"])
    
    st.markdown("---")
    
    # Tabs
    tabs = st.tabs(["📈 ANALYTICS", "💼 BUSINESS", "🌍 REGIONAL", "📊 DATA", "🔮 PREDICTIONS"])
    
    with tabs[0]:  # Analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📅 Energy Timeline")
            cat_ts = ts_df[ts_df["Category"] == category].copy()
            fig = px.line(cat_ts.head(90), x="Date", y="Energy_Usage",
                         title=f"{category} - 90-Day Pattern",
                         color_discrete_sequence=["#00C853"])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Performance Comparison")
            comp = {
                "Model": ["Legacy", "Modern", "Updated"],
                "Energy": [row["Old_Energy(W)"], row["Modern_Energy(W)"], row["Updated_Energy(W)"]],
                "Efficiency": [row["Old_Eff(%)"], row["Modern_Eff(%)"], row["Updated_Eff(%)"]]
            }
            fig = px.bar(comp, x="Model", y=["Energy", "Efficiency"], barmode="group",
                        color_discrete_sequence=["#00C853", "#64DD17"])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:  # Business
        st.markdown("### 💰 Financial Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **Investment Analysis**
            - Initial: ₹{row['Updated_Cost($)'] * 100:,.0f}
            - Annual Return: ₹{summary['annual_cost_saved']:,.0f}
            - Payback: {row['ROI_Months']} months
            - NPV (5yr): ₹{summary['annual_cost_saved'] * 5 - row['Updated_Cost($)'] * 100:,.0f}
            """)
        
        with col2:
            st.success(f"""
            **Market Position**
            - Features: {row['Smart_Features']}
            - Warranty: {row['Warranty_Years']} years
            - Score: {row['Maintenance_Score']}/100
            - Edge: High
            """)
        
        with col3:
            st.warning(f"""
            **Sales Projections**
            - Market: {region}
            - Adoption: 65%
            - Revenue: High
            - Competition: Moderate
            """)
    
    with tabs[2]:  # Regional
        st.markdown("### 🗺️ Regional Market Analysis")
        
        regional_filtered = regional_df[regional_df["Category"] == category]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(regional_filtered, x="Region", y="Adoption_Rate",
                        color="Adoption_Rate", 
                        color_continuous_scale=["#AEEA00", "#00C853"],
                        title="Adoption by Region")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(regional_filtered, x="Adoption_Rate", y="Avg_Savings",
                            size="Grid_Stability", color="Region",
                            title="Savings vs Adoption")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:  # Data
        if show_raw_data:
            st.markdown("### 📊 Product Specifications")
            st.dataframe(df, use_container_width=True)
            
            st.markdown("### 📈 Time Series (Last 30 Days)")
            st.dataframe(ts_df[ts_df["Category"] == category].tail(30), use_container_width=True)
            
            st.markdown("### 🗺️ Regional Statistics")
            st.dataframe(regional_df[regional_df["Category"] == category], use_container_width=True)
        else:
            st.info("Enable 'Show Raw Data' in sidebar to view detailed data tables")
    
    with tabs[4]:  # Predictions
        st.markdown("### 🔮 Market Predictions")
        
        years = np.arange(2024, 2034)
        adoption = [0.3 + (i * 0.06) for i in range(len(years))]
        revenue = [summary['annual_cost_saved'] * 1000 * (1.15 ** i) for i in range(len(years))]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(go.Scatter(x=years, y=adoption, 
                                       line=dict(color="#00C853", width=3),
                                       fill='tozeroy'))
            fig.update_layout(title="Adoption Forecast", yaxis_title="Rate")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(go.Scatter(x=years, y=revenue,
                                       line=dict(color="#64DD17", width=3),
                                       fill='tozeroy'))
            fig.update_layout(title="Revenue Projection", yaxis_title="₹")
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
# Footer
# ------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 20px; color: #666;'>
    <div style='color: #00C853; font-weight: 700; font-size: 1.2em; margin-bottom: 10px;'>
        🌿 ECOFUSION 3.0 - {st.session_state.view_mode.upper() if st.session_state.view_mode else 'PLATFORM'} VIEW
    </div>
    <div>© {datetime.now().year} EcoFusion | Making Sustainability Intelligent</div>
    <div style='font-size: 0.85em; margin-top: 10px;'>
        Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</div>
""", unsafe_allow_html=True)