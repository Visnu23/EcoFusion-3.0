# ======================================================
# app.py — EcoFusion 3.0 Streamlit Dashboard (Enhanced Premium Edition)
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
import altair as alt
import sys
import os

# Add model directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------
st.set_page_config(
    page_title="🌿 EcoFusion 3.0 – Smart Energy Intelligence",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# Enhanced Premium CSS with Gradients & Animations
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
        text-shadow: 0 2px 10px rgba(0, 255, 0, 0.2);
        letter-spacing: 1px;
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
        opacity: 0;
        animation: fadeInUp 1s ease 0.5s forwards;
        font-weight: 500;
        padding: 0 10%;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #0A0A0A, #111111);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 8px 8px 16px #000000, -8px -8px 16px #1A1A1A;
        border: 1px solid rgba(0, 255, 0, 0.15);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
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
        border-color: rgba(0, 255, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: 800;
        color: #00FF00;
        margin-bottom: 5px;
        text-shadow: 0 2px 4px rgba(0, 255, 0, 0.3);
    }
    
    .metric-label {
        color: #00CC00;
        font-size: 0.95em;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced Sidebar - DARK VERSION */
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
        box-shadow: 0 4px 12px rgba(0, 255, 0, 0.2);
    }
    
    /* Sidebar text and controls styling */
    .stSidebar .stSelectbox, 
    .stSidebar .stSlider, 
    .stSidebar .stButton,
    .stSidebar .stCheckbox,
    .stSidebar .stRadio,
    .stSidebar .stTextInput,
    .stSidebar .stNumberInput,
    .stSidebar .stTextArea {
        color: #00CC00 !important;
    }
    
    .stSidebar label {
        color: #00FF00 !important;
        font-weight: 600 !important;
    }
    
    .stSidebar .st-bd, 
    .stSidebar .st-at,
    .stSidebar .st-ae {
        color: #00CC00 !important;
    }
    
    /* Premium Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: linear-gradient(90deg, #111111, #1A1A1A);
        padding: 5px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #0A0A0A;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #00CC00;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #00FF00;
        background: #111111;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00AA00, #00CC00);
        color: #000000;
        box-shadow: 0 4px 12px rgba(0, 255, 0, 0.3);
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00AA00, #00FF00);
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00AA00 0%, #00CC00 100%);
        color: #000000;
        border: none;
        padding: 12px 28px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0, 255, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 255, 0, 0.4);
        background: linear-gradient(135deg, #00CC00 0%, #00FF00 100%);
    }
    
    .stButton > button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover::after {
        left: 100%;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00FF00;
        position: relative;
        padding-left: 20px;
    }
    
    h2::before, h3::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 6px;
        background: linear-gradient(180deg, #00AA00, #00FF00);
        border-radius: 3px;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #111111;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00AA00, #00FF00);
        border-radius: 4px;
    }
    
    /* Floating Elements */
    .floating {
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Glow Effects */
    .glow {
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
    }
    
    /* Status Indicators */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-green { background: #00FF00; box-shadow: 0 0 10px #00FF00; }
    .status-yellow { background: #FFD700; box-shadow: 0 0 10px #FFD700; }
    .status-red { background: #FF0000; box-shadow: 0 0 10px #FF0000; }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 12px;
        height: 12px;
        border: 2px solid #00FF00;
        border-top: 2px solid transparent;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(90deg, #0A0A0A, #111111);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00FF00;
        margin-top: 20px;
    }
    
    /* Dataframe Styling */
    .dataframe-container {
        background: #0A0A0A;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0, 255, 0, 0.1);
        margin: 10px 0;
        border: 1px solid #00AA00;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #00AA00;
        color: #000000;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #111111, #1A1A1A);
        color: #00FF00 !important;
        border: 1px solid #00AA00 !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #1A1A1A, #222222);
        border-color: #00FF00 !important;
    }
    
    /* Info, Success, Warning, Error boxes */
    .stAlert {
        background: #0A0A0A !important;
        border: 1px solid #00AA00 !important;
        color: #00CC00 !important;
    }
    
    .stAlert[data-baseweb="notification"] {
        background: #0A0A0A !important;
        border: 1px solid #00AA00 !important;
        color: #00CC00 !important;
    }
    
    /* Main background */
    .main .block-container {
        background-color: #000000;
    }
    
    /* Widget styling */
    .stSelectbox, .stSlider, .stCheckbox, .stRadio, .stTextInput, .stNumberInput {
        background: #0A0A0A !important;
        border: 1px solid #00AA00 !important;
        color: #00FF00 !important;
    }
    
    .stSelectbox:focus, .stSlider:focus, .stTextInput:focus, .stNumberInput:focus {
        border-color: #00FF00 !important;
        box-shadow: 0 0 0 1px rgba(0, 255, 0, 0.3) !important;
    }
    
    /* Checkboxes and radios */
    .stCheckbox label, .stRadio label {
        color: #00CC00 !important;
    }
    
    /* Metric display */
    .stMetric {
        background: #0A0A0A !important;
        border: 1px solid #00AA00 !important;
        color: #00FF00 !important;
    }
    
    .stMetric > div > div {
        color: #00FF00 !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: #0A0A0A !important;
        color: #00CC00 !important;
    }
    
    .dataframe th {
        background: #00AA00 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    .dataframe td {
        background: #111111 !important;
        color: #00CC00 !important;
        border: 1px solid #00AA00 !important;
    }
    
    .dataframe tr:hover {
        background: #1A1A1A !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: #0A0A0A !important;
        border: 1px solid #00AA00 !important;
        color: #00FF00 !important;
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly {
        background: #0A0A0A !important;
    }
    
    /* Streamlit text elements */
    .stText, .stMarkdown, .stHeader, .stSubheader {
        color: #00CC00 !important;
    }
    
    /* Input labels */
    .css-1q8dd3e {
        color: #00FF00 !important;
    }
    
    /* Remove Streamlit branding */
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
        
        # Modern improvements
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
            # Seasonal pattern
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            # Random noise
            noise = np.random.normal(0, 0.1)
            time_series.append({
                "Date": date,
                "Category": category,
                "Energy_Usage": round(base_usage * seasonal * (1 + noise), 2),
                "CO2_Emission": round(base_usage * seasonal * np.random.uniform(0.0004, 0.0006), 4),
                "Cost": round(base_usage * seasonal * np.random.uniform(0.001, 0.002), 2),
                "Temperature": np.random.uniform(15, 35) + 5 * np.sin(2 * np.pi * date.dayofyear / 365),
                "Peak_Hour": 1 if 17 <= date.hour <= 21 else 0
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

# ------------------------------------------------------
# Analysis Functions
# ------------------------------------------------------
def analyze_efficiency(category, hours_per_day, electricity_rate, df):
    row = df[df["Category"] == category].iloc[0]
    
    energy_diff = row["Old_Energy(W)"] - row["Updated_Energy(W)"]
    co2_diff = row["Old_CO2(kg/hr)"] - row["Updated_CO2(kg/hr)"]
    eff_gain = row["Updated_Eff(%)"] - row["Old_Eff(%)"]
    
    annual_energy_saved = energy_diff * hours_per_day * 365 / 1000
    annual_co2_saved = co2_diff * hours_per_day * 365
    annual_cost_saved = annual_energy_saved * electricity_rate
    
    # Generate component details
    components = {
        "Old": {
            "Motor": "Standard Induction",
            "Compressor": "Fixed Speed",
            "Control System": "Basic Thermostat",
            "Insulation": "Standard Foam",
            "Display": "Analog Dial"
        },
        "Modern": {
            "Motor": "Brushless DC",
            "Compressor": "Variable Speed",
            "Control System": "Digital Controller",
            "Insulation": "Enhanced Foam",
            "Display": "LED Digital"
        },
        "Updated": {
            "Motor": "AI-Optimized BLDC",
            "Compressor": "AI Adaptive Speed",
            "Control System": "IoT Smart Controller",
            "Insulation": "Vacuum Panel",
            "Display": "Touchscreen with Analytics"
        }
    }
    
    return {
        "old": f"Legacy {category}",
        "modern": f"Modern {category} Pro",
        "updated": f"Smart {category} X",
        "energy_diff": energy_diff,
        "co2_diff": co2_diff,
        "eff_gain": eff_gain,
        "annual_energy_saved": annual_energy_saved,
        "annual_co2_saved": annual_co2_saved,
        "annual_cost_saved": annual_cost_saved,
        "components": components,
        "row": row
    }

# ------------------------------------------------------
# Fixed highlight_best function
# ------------------------------------------------------
def highlight_best(val, col_name, comparison_data):
    """Highlight the best value in each column"""
    if col_name == 'Feature':
        return ''
    
    if 'W' in str(val) or 'kg' in str(val) or '$' in str(val) or '%' in str(val):
        # Extract numeric values
        legacy_val = comparison_data['Legacy'][comparison_data['Feature'].tolist().index(col_name)]
        modern_val = comparison_data['Modern'][comparison_data['Feature'].tolist().index(col_name)]
        updated_val = comparison_data['Self-Upgrading'][comparison_data['Feature'].tolist().index(col_name)]
        
        # Extract numbers from strings
        def extract_number(x):
            x_str = str(x)
            for char in ['W', 'kg', '$', '%', '/hr']:
                x_str = x_str.replace(char, '')
            try:
                return float(x_str)
            except:
                return 0
        
        numbers = [extract_number(legacy_val), extract_number(modern_val), extract_number(updated_val)]
        current_num = extract_number(val)
        
        # Determine if lower or higher is better
        if 'kg' in str(val) or 'W' in str(val) or '$' in str(val):
            best_val = min(numbers)
        else:
            best_val = max(numbers)
        
        if current_num == best_val:
            return 'background-color: #00C853; color: white; font-weight: bold; border-radius: 5px;'
    return ''

# ------------------------------------------------------
# Carbon Analysis Integration
# ------------------------------------------------------
def analyze_carbon_footprint(category, energy_consumption):
    """Analyze carbon footprint using model"""
    try:
        from model_enhanced import CO2IntelligenceModelEnhanced
        model = CO2IntelligenceModelEnhanced()
        
        # Map appliance to sector
        sector_map = {
            "Refrigerator": ("Buildings", "Commercial HVAC"),
            "Air Conditioner": ("Buildings", "Commercial HVAC"),
            "Washing Machine": ("Buildings", "Residential Heating"),
            "Television": ("Technology", "Data Center"),
            "Water Heater": ("Buildings", "Residential Heating"),
            "LED Lighting": ("Buildings", "Commercial HVAC"),
            "Computer": ("Technology", "Data Center"),
            "Microwave": ("Buildings", "Residential Heating"),
            "Dishwasher": ("Buildings", "Residential Heating"),
            "Electric Vehicle": ("Transportation", "Heavy Truck")
        }
        
        sector, cat = sector_map.get(category, ("Buildings", "Commercial HVAC"))
        
        # Calculate quantity based on energy consumption
        quantity = energy_consumption / 1000  # Approximate quantity
        
        analysis = model.analyze_carbon_reduction(
            sector=sector,
            category=cat,
            quantity=quantity,
            energy_consumption=energy_consumption,
            carbon_price=50.0
        )
        
        return analysis
        
    except Exception as e:
        st.warning(f"Carbon analysis not available: {str(e)}")
        return None

# ------------------------------------------------------
# Header with Animated Elements
# ------------------------------------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<div class='main-title'>🌿 ECOFUSION 3.0</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Advanced Sustainable Intelligence Platform for Energy Optimization & Carbon Footprint Reduction</div>", unsafe_allow_html=True)

# Animated separator
st.markdown(
    """
    <div style='text-align: center; margin: 30px 0;'>
        <div style='height: 3px; background: linear-gradient(90deg, transparent, #00C853, #64DD17, transparent); 
                    animation: expandWidth 2s ease forwards; width: 0%; margin: 0 auto; border-radius: 3px;'></div>
    </div>
    <style>
    @keyframes expandWidth {
        from { width: 0%; }
        to { width: 80%; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# Loading Animation with Progress
# ------------------------------------------------------
with st.spinner('🌱 Loading sustainable intelligence platform...'):
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
        progress_text.text(f"Initializing systems... {i+1}%")
    
    # Load synthetic data
    df, ts_df, regional_df = generate_synthetic_data()
    
    progress_text.text("✅ System ready! Data loaded successfully.")
    time.sleep(0.5)
    progress_text.empty()

# ------------------------------------------------------
# Enhanced Sidebar with Real-time Controls
# ------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>⚙️ CONTROL PANEL</div>", unsafe_allow_html=True)
    
    # Real-time metrics
    st.markdown("### 📊 Live Metrics")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Active Users", "1,247", "+12%")
    with col_s2:
        st.metric("CO₂ Saved", "2.4t", "Today")
    
    st.markdown("---")
    
    # Main controls
    category = st.selectbox(
        "🏷️ APPLIANCE CATEGORY",
        df["Category"].tolist(),
        index=0,
        help="Select appliance category for analysis"
    )
    
    st.markdown("### ⚡ USAGE PROFILE")
    
    col_s3, col_s4 = st.columns(2)
    with col_s3:
        hours_per_day = st.slider(
            "Daily Hours",
            1, 24, 8,
            help="Average daily usage hours"
        )
    with col_s4:
        electricity_rate = st.slider(
            "Rate (₹/kWh)",
            3, 15, 7,
            help="Local electricity rate"
        )
    
    # Usage pattern
    usage_pattern = st.select_slider(
        "🔄 Usage Pattern",
        options=["Light", "Moderate", "Heavy", "Commercial"],
        value="Moderate"
    )
    
    # Energy consumption estimate
    energy_consumption = st.slider(
        "⚡ Estimated Energy (kWh/month)",
        50, 2000, 300,
        help="Estimated monthly energy consumption"
    )
    
    # Regional selection
    region = st.selectbox(
        "📍 REGION",
        regional_df["Region"].unique(),
        help="Select your region for localized insights"
    )
    
    # Carbon analysis toggle
    enable_carbon_analysis = st.toggle("🌍 Enable Carbon Analysis", True)
    
    # Advanced features
    with st.expander("⚙️ ADVANCED SETTINGS"):
        ai_optimization = st.toggle("🤖 AI Optimization", True)
        real_time_monitoring = st.toggle("📡 Real-time Monitoring", True)
        predictive_maintenance = st.toggle("🔧 Predictive Maintenance", True)
        carbon_offset = st.toggle("🌳 Carbon Offset", False)
    
    # Action buttons
    st.markdown("---")
    col_s5, col_s6 = st.columns(2)
    with col_s5:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()
    with col_s6:
        if st.button("📥 Export Report", use_container_width=True):
            st.success("Report generated successfully!")
    
    # Live status
    st.markdown("---")
    st.markdown("### 📶 SYSTEM STATUS")
    st.markdown('<span class="status-dot status-green"></span> Live Data Stream', unsafe_allow_html=True)
    st.markdown('<span class="status-dot status-green"></span> AI Models Active', unsafe_allow_html=True)
    st.markdown('<span class="status-dot status-yellow"></span> Optimization Running', unsafe_allow_html=True)

# ------------------------------------------------------
# Analysis
# ------------------------------------------------------
summary = analyze_efficiency(category, hours_per_day, electricity_rate, df)
row = summary["row"]

# Carbon analysis
carbon_analysis = None
if enable_carbon_analysis:
    with st.spinner("🌱 Analyzing carbon footprint..."):
        carbon_analysis = analyze_carbon_footprint(category, energy_consumption)

# ------------------------------------------------------
# Premium Metrics Dashboard
# ------------------------------------------------------
st.subheader(f"📊 {category} PERFORMANCE DASHBOARD")

# Create animated metric cards
cols = st.columns(4)
metric_configs = [
    (f"{summary['energy_diff']:.1f} W", "⚡ ENERGY REDUCTION", "#00C853", "energy"),
    (f"{summary['annual_co2_saved']:.1f} kg/yr", "🌍 CO₂ SAVINGS", "#64DD17", "co2"),
    (f"₹{summary['annual_cost_saved']:.0f}/yr", "💰 ANNUAL SAVINGS", "#AEEA00", "cost"),
    (f"+{summary['eff_gain']:.1f}%", "📈 EFFICIENCY GAIN", "#00E676", "efficiency")
]

for idx, (value, label, color, icon) in enumerate(metric_configs):
    with cols[idx]:
        st.markdown(
            f"""
            <div class='metric-card floating' style='border-top-color: {color};'>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>
                    <span style='color: {color}; font-size: 1.4em;'>{icon}</span>
                    {label}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ROI and Payback Analysis
col_r1, col_r2, col_r3, col_r4 = st.columns(4)
with col_r1:
    roi_months = row["ROI_Months"]
    st.metric("📅 ROI Period", f"{roi_months} months", f"-{roi_months//12} years")
with col_r2:
    maintenance = row["Maintenance_Score"]
    st.metric("🔧 Maintenance Score", f"{maintenance}/100", f"+{maintenance-70}%")
with col_r3:
    warranty = row["Warranty_Years"]
    st.metric("🛡️ Warranty", f"{warranty} years", "Extended" if warranty > 2 else "Standard")
with col_r4:
    smart_level = row["Smart_Features"]
    st.metric("🤖 Smart Level", smart_level, "AI Enabled" if smart_level == "Advanced" else "")

st.markdown("---")

# ------------------------------------------------------
# Interactive Tabs System
# ------------------------------------------------------
tabs = st.tabs(["📈 ANALYTICS", "🌍 IMPACT", "💰 FINANCE", "📊 COMPARISON", "🔮 PREDICTIONS", "🌱 CARBON IQ"])

with tabs[0]:  # Analytics Tab
    col_a1, col_a2 = st.columns([3, 2])
    
    with col_a1:
        # Energy Timeline Chart
        st.markdown("### 📅 ENERGY CONSUMPTION TIMELINE")
        
        # Filter time series for selected category
        cat_ts = ts_df[ts_df["Category"] == category].copy()
        cat_ts["Date"] = pd.to_datetime(cat_ts["Date"])
        
        fig_timeline = px.line(cat_ts.head(30), x="Date", y="Energy_Usage",
                              title=f"{category} - Daily Energy Consumption",
                              color_discrete_sequence=["#00C853"])
        fig_timeline.update_layout(
            plot_bgcolor='rgba(240, 248, 240, 0.5)',
            paper_bgcolor='rgba(255,255,255,0)',
            height=400
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col_a2:
        # Efficiency Radar
        st.markdown("### 🎯 EFFICIENCY RADAR")
        
        metrics = ["Energy", "CO₂", "Cost", "Reliability", "Smart Features"]
        values = [
            row["Updated_Eff(%)"],
            100 - (row["Updated_CO2(kg/hr)"] / row["Old_CO2(kg/hr)"] * 100),
            100 - (row["Updated_Cost($)"] / row["Old_Cost($)"] * 100),
            row["Maintenance_Score"],
            90 if row["Smart_Features"] == "Advanced" else 70 if row["Smart_Features"] == "Moderate" else 50
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            fillcolor='rgba(0, 200, 83, 0.3)',
            line=dict(color='#00C853', width=2)
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
                bgcolor='rgba(240, 248, 240, 0.5)'
            ),
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Regional Comparison
    st.markdown("### 🌐 REGIONAL ADOPTION ANALYSIS")
    regional_filtered = regional_df[regional_df["Category"] == category]
    
    col_a3, col_a4, col_a5 = st.columns(3)
    with col_a3:
        fig_regional = px.bar(regional_filtered, x="Region", y="Adoption_Rate",
                             color="Adoption_Rate",
                             color_continuous_scale=["#AEEA00", "#64DD17", "#00C853"],
                             title="Adoption Rate by Region")
        fig_regional.update_layout(height=300)
        st.plotly_chart(fig_regional, use_container_width=True)
    
    with col_a4:
        fig_savings = px.scatter(regional_filtered, x="Adoption_Rate", y="Avg_Savings",
                                size="Grid_Stability", color="Region",
                                color_discrete_sequence=["#00C853", "#64DD17", "#AEEA00", "#00E676", "#76FF03"],
                                title="Savings vs Adoption")
        fig_savings.update_layout(height=300)
        st.plotly_chart(fig_savings, use_container_width=True)
    
    with col_a5:
        # Heatmap data
        heat_data = []
        for reg in regional_df["Region"].unique():
            for cat in df["Category"].unique():
                filtered = regional_df[(regional_df["Region"] == reg) & (regional_df["Category"] == cat)]
                if not filtered.empty:
                    heat_data.append({
                        "Region": reg,
                        "Category": cat,
                        "Value": filtered["Adoption_Rate"].values[0]
                    })
        
        heat_df = pd.DataFrame(heat_data)
        fig_heat = px.density_heatmap(heat_df, x="Region", y="Category", z="Value",
                                     color_continuous_scale=["#E8F5E9", "#AEEA00", "#64DD17", "#00C853"],
                                     title="Regional Adoption Heatmap")
        fig_heat.update_layout(height=300)
        st.plotly_chart(fig_heat, use_container_width=True)

with tabs[1]:  # Impact Tab
    col_i1, col_i2 = st.columns([2, 1])
    
    with col_i1:
        # CO2 Impact Visualization
        st.markdown("### 🌍 ENVIRONMENTAL IMPACT")
        
        co2_data = {
            "Model": ["Legacy", "Modern", "Self-Upgrading"],
            "Daily_CO2": [
                row["Old_CO2(kg/hr)"] * hours_per_day,
                row["Modern_CO2(kg/hr)"] * hours_per_day,
                row["Updated_CO2(kg/hr)"] * hours_per_day
            ],
            "Annual_CO2": [
                row["Old_CO2(kg/hr)"] * hours_per_day * 365,
                row["Modern_CO2(kg/hr)"] * hours_per_day * 365,
                row["Updated_CO2(kg/hr)"] * hours_per_day * 365
            ]
        }
        
        fig_co2 = px.bar(co2_data, x="Model", y="Annual_CO2",
                        color="Model",
                        color_discrete_map={
                            "Legacy": "#ef5350",
                            "Modern": "#fbc02d",
                            "Self-Upgrading": "#00C853"
                        },
                        text=[f"{x:.1f} kg" for x in co2_data["Annual_CO2"]])
        fig_co2.update_layout(
            title="Annual CO₂ Emissions Comparison",
            height=400,
            plot_bgcolor='rgba(240, 248, 240, 0.5)'
        )
        st.plotly_chart(fig_co2, use_container_width=True)
    
    with col_i2:
        # Environmental Equivalents
        st.markdown("### 🌳 ENVIRONMENTAL EQUIVALENTS")
        
        co2_saved = summary["annual_co2_saved"]
        
        equivalents = [
            ("🌳 Trees Equivalent", f"{co2_saved/21:.1f} trees planted"),
            ("🚗 Car Miles", f"Equivalent to {co2_saved/0.4:.0f} miles not driven"),
            ("🏠 Home Energy", f"{co2_saved/4:.1f} months of home energy"),
            ("✈️ Flight Hours", f"{co2_saved/90:.1f} hours of flight emissions"),
            ("📱 Smartphones", f"Production of {co2_saved/16:.0f} smartphones")
        ]
        
        for icon, text in equivalents:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #E8F5E9, #F1F8E9); 
                        padding: 15px; border-radius: 10px; margin-bottom: 10px;
                        border-left: 4px solid #00C853;'>
                <div style='font-weight: 600; color: #1B5E20; margin-bottom: 5px;'>{icon}</div>
                <div style='color: #388E3C; font-size: 0.9em;'>{text}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Impact Timeline
    st.markdown("### 📅 10-YEAR IMPACT PROJECTION")
    
    years = list(range(1, 11))
    cumulative_savings = []
    cumulative_co2 = []
    cumulative_cost = []
    
    for year in years:
        cumulative_savings.append(summary["annual_energy_saved"] * year)
        cumulative_co2.append(summary["annual_co2_saved"] * year)
        cumulative_cost.append(summary["annual_cost_saved"] * year)
    
    fig_projection = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Energy Savings (kWh)", "CO₂ Reduction (kg)", "Cost Savings (₹)")
    )
    
    fig_projection.add_trace(
        go.Scatter(x=years, y=cumulative_savings, name="Energy", line=dict(color="#00C853", width=3)),
        row=1, col=1
    )
    
    fig_projection.add_trace(
        go.Scatter(x=years, y=cumulative_co2, name="CO₂", line=dict(color="#64DD17", width=3)),
        row=1, col=2
    )
    
    fig_projection.add_trace(
        go.Scatter(x=years, y=cumulative_cost, name="Cost", line=dict(color="#AEEA00", width=3)),
        row=1, col=3
    )
    
    fig_projection.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_projection, use_container_width=True)

with tabs[2]:  # Finance Tab
    col_f1, col_f2 = st.columns([3, 2])
    
    with col_f1:
        # Cost Analysis
        st.markdown("### 💰 COST-BENEFIT ANALYSIS")
        
        cost_data = {
            "Model": ["Legacy", "Modern", "Self-Upgrading"],
            "Initial_Cost": [
                row["Old_Cost($)"] * 100,
                row["Modern_Cost($)"] * 100,
                row["Updated_Cost($)"] * 100
            ],
            "Annual_Cost": [
                (row["Old_Energy(W)"] * hours_per_day * 365 / 1000) * electricity_rate,
                (row["Modern_Energy(W)"] * hours_per_day * 365 / 1000) * electricity_rate,
                (row["Updated_Energy(W)"] * hours_per_day * 365 / 1000) * electricity_rate
            ],
            "5_Year_Total": [
                (row["Old_Cost($)"] * 100) + ((row["Old_Energy(W)"] * hours_per_day * 365 / 1000) * electricity_rate * 5),
                (row["Modern_Cost($)"] * 100) + ((row["Modern_Energy(W)"] * hours_per_day * 365 / 1000) * electricity_rate * 5),
                (row["Updated_Cost($)"] * 100) + ((row["Updated_Energy(W)"] * hours_per_day * 365 / 1000) * electricity_rate * 5)
            ]
        }
        
        fig_cost = px.line(cost_data, x="Model", y="5_Year_Total",
                          markers=True,
                          title="5-Year Total Cost of Ownership",
                          color_discrete_sequence=["#00C853"])
        fig_cost.add_bar(x=cost_data["Model"], y=cost_data["Initial_Cost"],
                        name="Initial Cost",
                        marker_color="#AEEA00",
                        opacity=0.6)
        fig_cost.update_layout(height=400)
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col_f2:
        # Financial Metrics
        st.markdown("### 📊 FINANCIAL METRICS")
        
        metrics = [
            ("💰 Payback Period", f"{roi_months} months", "#00C853"),
            ("📈 NPV (5 years)", f"₹{summary['annual_cost_saved'] * 5 - row['Updated_Cost($)'] * 100:,.0f}", "#64DD17"),
            ("🎯 IRR", f"{min(50, (summary['annual_cost_saved'] / (row['Updated_Cost($)'] * 100)) * 100):.1f}%", "#AEEA00"),
            ("🔄 Break-even", f"Year {max(1, roi_months // 12)}", "#00E676"),
            ("🏆 Savings Ratio", f"1:{summary['annual_cost_saved'] / (row['Updated_Cost($)'] * 100):.1f}", "#76FF03")
        ]
        
        for label, value, color in metrics:
            st.markdown(f"""
            <div style='background: white; border-radius: 10px; padding: 15px; margin-bottom: 10px;
                        border: 2px solid {color}; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
                <div style='font-size: 0.85em; color: #666; font-weight: 600;'>{label}</div>
                <div style='font-size: 1.4em; font-weight: 700; color: {color}; margin-top: 5px;'>{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Incentives and Rebates
    st.markdown("### 🎁 AVAILABLE INCENTIVES")
    
    col_f3, col_f4, col_f5 = st.columns(3)
    
    with col_f3:
        st.info(f"""
        **Government Rebate**
        - Up to ₹{int(summary['annual_cost_saved'] * 0.3):,} rebate
        - Tax credit available
        - 5-year warranty included
        """)
    
    with col_f4:
        st.success(f"""
        **Utility Company Incentives**
        - ₹{int(summary['annual_cost_saved'] * 0.2):,} cashback
        - Lower electricity rates
        - Free energy audit
        """)
    
    with col_f5:
        st.warning(f"""
        **Manufacturer Offers**
        - 0% financing available
        - Free installation
        - Extended warranty
        - Trade-in bonus
        """)

with tabs[3]:  # Comparison Tab
    st.markdown("### 🔍 SIDE-BY-SIDE COMPARISON")
    
    # Comparison matrix - FIXED VERSION
    comparison_data = {
        "Feature": [
            "Energy Consumption (W)",
            "CO₂ Emissions (kg/hr)",
            "Efficiency (%)",
            "Smart Features",
            "Maintenance Score",
            "Warranty (Years)",
            "ROI Period (Months)",
            "Initial Cost ($)"
        ],
        "Legacy": [
            f"{row['Old_Energy(W)']:.1f}",
            f"{row['Old_CO2(kg/hr)']:.4f}",
            f"{row['Old_Eff(%)']:.1f}%",
            "Basic",
            f"{max(0, row['Maintenance_Score'] - 20)}/100",
            "1",
            "N/A",
            f"${row['Old_Cost($)'] * 100:.0f}"
        ],
        "Modern": [
            f"{row['Modern_Energy(W)']:.1f}",
            f"{row['Modern_CO2(kg/hr)']:.4f}",
            f"{row['Modern_Eff(%)']:.1f}%",
            row["Smart_Features"],
            f"{row['Maintenance_Score']}/100",
            f"{min(3, row['Warranty_Years'])}",
            f"{row['ROI_Months'] + 12}",
            f"${row['Modern_Cost($)'] * 100:.0f}"
        ],
        "Self-Upgrading": [
            f"{row['Updated_Energy(W)']:.1f}",
            f"{row['Updated_CO2(kg/hr)']:.4f}",
            f"{row['Updated_Eff(%)']:.1f}%",
            "Advanced",
            f"{min(100, row['Maintenance_Score'] + 10)}/100",
            f"{row['Warranty_Years']}",
            f"{row['ROI_Months']}",
            f"${row['Updated_Cost($)'] * 100:.0f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Apply styling - FIXED VERSION
    def highlight_best_wrapper(df):
        """Wrapper function to apply styling"""
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        
        for col in df.columns:
            if col == 'Feature':
                continue
            
            for idx in df.index:
                val = df.at[idx, col]
                col_name = df.at[idx, 'Feature']
                if 'W' in str(val) or 'kg' in str(val) or '$' in str(val) or '%' in str(val):
                    # Extract numeric values
                    def extract_number(x):
                        x_str = str(x)
                        for char in ['W', 'kg', '$', '%', '/hr', '/100']:
                            x_str = x_str.replace(char, '')
                        try:
                            return float(x_str)
                        except:
                            return 0
                    
                    legacy_val = df.at[df[df['Feature'] == col_name].index[0], 'Legacy']
                    modern_val = df.at[df[df['Feature'] == col_name].index[0], 'Modern']
                    updated_val = df.at[df[df['Feature'] == col_name].index[0], 'Self-Upgrading']
                    
                    numbers = [extract_number(legacy_val), extract_number(modern_val), extract_number(updated_val)]
                    current_num = extract_number(val)
                    
                    # Determine if lower or higher is better
                    if 'kg' in str(val) or 'W' in str(val) or '$' in str(val):
                        best_val = min(numbers)
                    else:
                        best_val = max(numbers)
                    
                    if current_num == best_val:
                        styles.at[idx, col] = 'background-color: #00C853; color: white; font-weight: bold; border-radius: 5px;'
        
        return styles
    
    # Apply the styling
    styled_df = comparison_df.style.apply(highlight_best_wrapper, axis=None)
    
    # Display the styled dataframe
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Component Evolution
    st.markdown("### 🔧 COMPONENT EVOLUTION")
    
    components = summary["components"]
    if components:
        all_keys = set(components["Old"].keys()) | set(components["Modern"].keys()) | set(components["Updated"].keys())
        
        comp_cols = st.columns(3)
        
        for idx, (title, data_dict) in enumerate([("Legacy", components["Old"]), 
                                                  ("Modern", components["Modern"]), 
                                                  ("Self-Upgrading", components["Updated"])]):
            with comp_cols[idx]:
                st.markdown(f"##### {title}")
                for key in all_keys:
                    value = data_dict.get(key, "—")
                    st.markdown(f"""
                    <div style='background: {"#E8F5E9" if idx == 2 else "#F5F5F5" if idx == 1 else "#FFEBEE"};
                                padding: 10px; margin: 5px 0; border-radius: 8px;
                                border-left: 4px solid {"#00C853" if idx == 2 else "#FFC107" if idx == 1 else "#EF5350"};'>
                        <div style='font-weight: 600; color: #333;'>{key}</div>
                        <div style='color: #666; font-size: 0.9em;'>{value}</div>
                    </div>
                    """, unsafe_allow_html=True)

with tabs[4]:  # Predictions Tab
    st.markdown("### 🔮 AI-POWERED PREDICTIONS")
    
    col_p1, col_p2 = st.columns([2, 1])
    
    with col_p1:
        # Future predictions
        years = np.arange(2024, 2034)
        
        # Generate future predictions
        future_energy = [row["Updated_Energy(W)"] * (0.98 ** (i)) for i in range(len(years))]
        future_co2 = [row["Updated_CO2(kg/hr)"] * (0.97 ** (i)) for i in range(len(years))]
        future_cost = [row["Updated_Cost($)"] * 100 * (0.95 ** (i)) for i in range(len(years))]
        future_efficiency = [min(100, row["Updated_Eff(%)"] + i * 2) for i in range(len(years))]
        
        fig_future = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Energy Consumption Forecast", "CO₂ Emissions Forecast",
                           "Cost Projection", "Efficiency Improvement")
        )
        
        fig_future.add_trace(
            go.Scatter(x=years, y=future_energy, name="Energy", line=dict(color="#00C853", width=3)),
            row=1, col=1
        )
        
        fig_future.add_trace(
            go.Scatter(x=years, y=future_co2, name="CO₂", line=dict(color="#64DD17", width=3)),
            row=1, col=2
        )
        
        fig_future.add_trace(
            go.Scatter(x=years, y=future_cost, name="Cost", line=dict(color="#AEEA00", width=3)),
            row=2, col=1
        )
        
        fig_future.add_trace(
            go.Scatter(x=years, y=future_efficiency, name="Efficiency", line=dict(color="#00E676", width=3)),
            row=2, col=2
        )
        
        fig_future.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_future, use_container_width=True)
    
    with col_p2:
        # AI Recommendations
        st.markdown("### 🤖 AI RECOMMENDATIONS")
        
        eco_score = (summary["eff_gain"] * 3) + (summary["annual_co2_saved"] / 10) + (summary["annual_cost_saved"] / 100)
        
        if eco_score > 120:
            recommendation = "🚀 IMMEDIATE UPGRADE"
            color = "#00C853"
            confidence = "95%"
            actions = [
                "Schedule installation this month",
                "Apply for government rebates",
                "Set up smart monitoring",
                "Join energy savings program"
            ]
        elif eco_score > 80:
            recommendation = "📅 PLAN UPGRADE"
            color = "#FFC107"
            confidence = "75%"
            actions = [
                "Research available models",
                "Save for investment",
                "Monitor energy prices",
                "Wait for seasonal discounts"
            ]
        else:
            recommendation = "⏳ MAINTAIN & OPTIMIZE"
            color = "#EF5350"
            confidence = "60%"
            actions = [
                "Optimize current usage",
                "Regular maintenance",
                "Monitor for price drops",
                "Consider partial upgrades"
            ]
        
        st.markdown(f"""
        <div style='background: {color}; color: white; padding: 20px; border-radius: 15px; 
                    text-align: center; margin-bottom: 20px;'>
            <div style='font-size: 1.2em; font-weight: 700;'>{recommendation}</div>
            <div style='font-size: 0.9em; opacity: 0.9;'>AI Confidence: {confidence}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Recommended Actions:**")
        for action in actions:
            st.markdown(f"✅ {action}")
        
        # Savings potential
        potential_savings = {
            "Next 3 years": f"₹{summary['annual_cost_saved'] * 3:,.0f}",
            "Next 5 years": f"₹{summary['annual_cost_saved'] * 5:,.0f}",
            "Lifetime (10y)": f"₹{summary['annual_cost_saved'] * 10:,.0f}"
        }
        
        st.markdown("---")
        st.markdown("**💰 Savings Potential:**")
        for period, amount in potential_savings.items():
            st.markdown(f"{period}: **{amount}**")

with tabs[5]:  # Carbon IQ Tab
    st.markdown("### 🌱 CARBON INTELLIGENCE DASHBOARD")
    
    if carbon_analysis and carbon_analysis.get('success'):
        carbon_data = carbon_analysis
        
        # Carbon Metrics
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        
        with col_c1:
            current_co2 = carbon_data.get('carbon_metrics', {}).get('current_carbon_footprint_tons', 0)
            st.metric("Current Footprint", f"{current_co2:.1f} tons", "CO₂/year")
        
        with col_c2:
            reduction_pct = carbon_data.get('carbon_metrics', {}).get('reduction_percentage', 0)
            st.metric("Reduction Potential", f"{reduction_pct:.1f}%", "vs Baseline")
        
        with col_c3:
            credits = carbon_data.get('carbon_metrics', {}).get('carbon_credits_generated', 0)
            st.metric("Carbon Credits", f"{credits:.1f}", "tons/year")
        
        with col_c4:
            neutral_year = carbon_data.get('carbon_metrics', {}).get('carbon_neutral_timeline_years', 15)
            st.metric("Carbon Neutral", f"Year {neutral_year}", "Projected")
        
        # Carbon Analysis Details
        col_c5, col_c6 = st.columns([2, 1])
        
        with col_c5:
            st.markdown("#### 📊 Carbon Reduction Analysis")
            if 'quantum_optimization' in carbon_data:
                qo = carbon_data['quantum_optimization']
                if 'best_solution' in qo:
                    solution = qo['best_solution']
                    st.write(f"**Best Strategy:** {solution.get('strategy_type', 'N/A')}")
                    st.write(f"**CO₂ Reduction:** {solution.get('co2_reduction_pct', 0):.1f}%")
                    st.write(f"**Implementation Cost:** ₹{solution.get('implementation_cost', 0):,.0f}")
                    st.write(f"**Payback Period:** {solution.get('payback_period', 0):.1f} years")
        
        with col_c6:
            st.markdown("#### 🎯 Carbon Score")
            carbon_score = carbon_data.get('carbon_metrics', {}).get('carbon_score', 0)
            
            # Create gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=carbon_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Carbon Intelligence"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00C853"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ef5350"},
                        {'range': [50, 75], 'color': "#fbc02d"},
                        {'range': [75, 100], 'color': "#00C853"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': carbon_score
                    }
                }
            ))
            
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Environmental Benefits
        st.markdown("#### 🌍 Environmental Benefits")
        if 'environmental_impact' in carbon_data:
            env = carbon_data['environmental_impact']
            col_c7, col_c8, col_c9 = st.columns(3)
            
            with col_c7:
                st.metric("Trees Equivalent", f"{env.get('equivalent_trees_planted', 0):.0f}", "trees")
            
            with col_c8:
                st.metric("Cars Removed", f"{env.get('equivalent_cars_removed', 0):.0f}", "vehicles")
            
            with col_c9:
                st.metric("Health Savings", f"₹{env.get('health_benefits', {}).get('healthcare_cost_savings', 0):,.0f}", "annually")
        
        # Financial Analysis
        st.markdown("#### 💰 Financial Impact")
        if 'financial_analysis' in carbon_data:
            fin = carbon_data['financial_analysis']
            
            col_c10, col_c11, col_c12, col_c13 = st.columns(4)
            
            with col_c10:
                st.metric("Annual Savings", f"₹{fin.get('annual_total_savings', 0):,.0f}", "Total")
            
            with col_c11:
                st.metric("NPV (10yr)", f"₹{fin.get('net_present_value', 0):,.0f}", "Net Present Value")
            
            with col_c12:
                st.metric("IRR", f"{fin.get('internal_rate_of_return', 0):.1f}%", "Return")
            
            with col_c13:
                st.metric("Break-even", f"Year {fin.get('break_even_year', 0)}", "Payback")
        
    else:
        st.warning("Carbon analysis is not available. Please enable it in the sidebar settings.")
        if carbon_analysis and 'error' in carbon_analysis:
            st.error(f"Error: {carbon_analysis['error']}")

# ------------------------------------------------------
# Bottom Action Bar
# ------------------------------------------------------
st.markdown("---")
col_b1, col_b2, col_b3 = st.columns([2, 1, 1])

with col_b1:
    st.markdown("### 📋 QUICK ACTIONS")
    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("📧 Contact Expert", use_container_width=True):
            st.success("Expert contact information sent!")
    with action_cols[1]:
        if st.button("📊 Download Report", use_container_width=True):
            st.success("Report downloading...")
    with action_cols[2]:
        if st.button("🔄 Compare Another", use_container_width=True):
            st.rerun()
    with action_cols[3]:
        if st.button("⭐ Save Analysis", use_container_width=True):
            st.success("Analysis saved to your dashboard!")

with col_b2:
    st.markdown("### 📶 LIVE STATUS")
    st.markdown(f"""
    <div style='background: #E8F5E9; padding: 15px; border-radius: 10px;'>
        <div style='display: flex; align-items: center; margin-bottom: 10px;'>
            <span class='status-dot status-green'></span>
            <span style='font-weight: 600;'>System: Online</span>
        </div>
        <div style='display: flex; align-items: center; margin-bottom: 10px;'>
            <span class='status-dot status-green'></span>
            <span style='font-weight: 600;'>Data: Streaming</span>
        </div>
        <div style='display: flex; align-items: center;'>
            <span class='status-dot status-yellow'></span>
            <span style='font-weight: 600;'>AI: Optimizing</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_b3:
    st.markdown("### ⚡ QUICK STATS")
    st.metric("Analysis Run", "1,247", "+12 today")
    st.metric("CO₂ Reduced", "2.4 tons", "+0.1 today")
    st.metric("Savings Generated", "₹1.2M", "+₹12K today")

# ------------------------------------------------------
# Animated Footer with Live Updates
# ------------------------------------------------------
st.markdown("---")
footer_cols = st.columns([2, 1, 1])

with footer_cols[0]:
    st.markdown(f"""
    <div class='footer'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
            <div>
                <span style='color: #00C853; font-weight: 700; font-size: 1.1em;'>🌿 ECOFUSION 3.0</span>
                <span style='color: #666; margin-left: 10px;'>Premium Sustainable Intelligence</span>
            </div>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <span class='loading-spinner'></span>
                <span style='color: #00C853; font-weight: 600;'>Live</span>
            </div>
        </div>
        <div style='color: #666; font-size: 0.85em; line-height: 1.4;'>
            © {datetime.now().year} EcoFusion 3.0 | Developed for IBM Z Datathon | 
            <span style='color: #00C853;'>Making Sustainability Intelligent</span>
        </div>
        <div style='color: #888; font-size: 0.75em; margin-top: 10px;'>
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
            Data refresh: Every 15 minutes
        </div>
    </div>
    """, unsafe_allow_html=True)

with footer_cols[1]:
    # Real-time carbon offset
    offset = summary["annual_co2_saved"] / 365 * (datetime.now().hour / 24)
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #E8F5E9, #C8E6C9); 
                padding: 15px; border-radius: 10px; text-align: center;'>
        <div style='color: #1B5E20; font-weight: 600; margin-bottom: 5px;'>🌳 Today's Offset</div>
        <div style='color: #00C853; font-size: 1.3em; font-weight: 700;'>{offset:.3f} kg CO₂</div>
        <div style='color: #388E3C; font-size: 0.8em;'>Equivalent to {offset/0.021:.1f} tree hours</div>
    </div>
    """, unsafe_allow_html=True)

with footer_cols[2]:
    # Energy savings counter
    energy_saved = summary["annual_energy_saved"] / 365 * (datetime.now().hour / 24)
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #F1F8E9, #DCEDC8); 
                padding: 15px; border-radius: 10px; text-align: center;'>
        <div style='color: #1B5E20; font-weight: 600; margin-bottom: 5px;'>⚡ Energy Saved</div>
        <div style='color: #64DD17; font-size: 1.3em; font-weight: 700;'>{energy_saved:.1f} kWh</div>
        <div style='color: #388E3C; font-size: 0.8em;'>Powering {energy_saved/1.5:.0f} homes for 1 hour</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------
# Auto-refresh notification
# ------------------------------------------------------
refresh_placeholder = st.empty()
if st.button("🔄 LIVE REFRESH", key="final_refresh", type="primary"):
    with refresh_placeholder:
        st.balloons()
        st.success("Dashboard refreshed with latest data!")
        time.sleep(1)
        st.rerun()

# ------------------------------------------------------
# Performance Tips (Always visible)
# ------------------------------------------------------
with st.expander("💡 PERFORMANCE TIPS & BEST PRACTICES", expanded=False):
    tip_cols = st.columns(3)
    
    with tip_cols[0]:
        st.markdown("""
        **⚡ Energy Optimization**
        - Use during off-peak hours
        - Maintain optimal temperatures
        - Regular cleaning and maintenance
        - Enable sleep modes when idle
        """)
    
    with tip_cols[1]:
        st.markdown("""
        **💰 Cost Savings**
        - Leverage government rebates
        - Consider time-of-use rates
        - Bundle with other upgrades
        - Monitor usage patterns
        """)
    
    with tip_cols[2]:
        st.markdown("""
        **🌍 Environmental Impact**
        - Join carbon offset programs
        - Recycle old appliances properly
        - Choose ENERGY STAR certified
        - Share savings with community
        """)

# ------------------------------------------------------
# Live Data Simulation (Optional)
# ------------------------------------------------------
if st.checkbox("📡 Show Live Data Stream", False):
    placeholder = st.empty()
    
    for seconds in range(10):
        with placeholder.container():
            st.markdown("### 📊 LIVE DATA STREAM")
            live_cols = st.columns(4)
            
            with live_cols[0]:
                st.metric("Current Power", f"{row['Updated_Energy(W)'] * np.random.uniform(0.9, 1.1):.1f}W", 
                         f"{np.random.choice(['-', '+'])}{np.random.uniform(0.1, 0.5):.1f}W")
            
            with live_cols[1]:
                st.metric("CO₂ Emission", f"{row['Updated_CO2(kg/hr)'] * np.random.uniform(0.9, 1.1):.4f}kg/hr",
                         f"{np.random.choice(['-', '+'])}{np.random.uniform(0.0001, 0.0003):.4f}kg/hr")
            
            with live_cols[2]:
                st.metric("Efficiency", f"{row['Updated_Eff(%)'] * np.random.uniform(0.99, 1.01):.1f}%",
                         f"{np.random.choice(['-', '+'])}{np.random.uniform(0.1, 0.3):.1f}%")
            
            with live_cols[3]:
                st.metric("Current Cost", f"₹{(row['Updated_Energy(W)'] * electricity_rate / 1000) * np.random.uniform(0.9, 1.1):.2f}/hr",
                         f"{np.random.choice(['-₹', '+₹'])}{np.random.uniform(0.01, 0.05):.2f}/hr")
            
            time.sleep(1)
    
    placeholder.empty()

# ------------------------------------------------------
# Session Statistics
# ------------------------------------------------------
if 'visit_count' not in st.session_state:
    st.session_state.visit_count = 0
st.session_state.visit_count += 1

# Hidden debug information
with st.expander("🔧 Debug Information", expanded=False):
    st.write(f"Session visits: {st.session_state.visit_count}")
    st.write(f"Data shape: {df.shape}")
    st.write(f"Selected category: {category}")
    st.write(f"Carbon analysis available: {carbon_analysis is not None}")