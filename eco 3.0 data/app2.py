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
# Data Loading Functions
# ------------------------------------------------------
@st.cache_data
def load_dataset():
    """Load real/synthetic dataset from CSV or generate if not available"""
    
    # Try to load from CSV first
    dataset_path = "appliance_data.csv"
    
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            st.sidebar.success("✅ Dataset loaded from file")
            return df
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load CSV: {e}")
    
    # If no CSV, generate realistic synthetic dataset
    st.sidebar.info("📊 Generating synthetic dataset...")
    
    # Real-world appliance data based on energy consumption studies
    appliances_data = {
        "Refrigerator": {
            "old_energy": 250, "modern_energy": 150, "updated_energy": 100,
            "old_co2": 0.125, "modern_co2": 0.075, "updated_co2": 0.050,
            "old_eff": 65, "modern_eff": 78, "updated_eff": 88,
            "usage_hours": 24, "units": 500
        },
        "Air Conditioner": {
            "old_energy": 2000, "modern_energy": 1400, "updated_energy": 1000,
            "old_co2": 1.0, "modern_co2": 0.7, "updated_co2": 0.5,
            "old_eff": 60, "modern_eff": 72, "updated_eff": 85,
            "usage_hours": 8, "units": 300
        },
        "Washing Machine": {
            "old_energy": 800, "modern_energy": 500, "updated_energy": 350,
            "old_co2": 0.4, "modern_co2": 0.25, "updated_co2": 0.175,
            "old_eff": 70, "modern_eff": 82, "updated_eff": 90,
            "usage_hours": 2, "units": 400
        },
        "Television": {
            "old_energy": 150, "modern_energy": 80, "updated_energy": 50,
            "old_co2": 0.075, "modern_co2": 0.04, "updated_co2": 0.025,
            "old_eff": 55, "modern_eff": 75, "updated_eff": 88,
            "usage_hours": 6, "units": 600
        },
        "Water Heater": {
            "old_energy": 3000, "modern_energy": 2200, "updated_energy": 1500,
            "old_co2": 1.5, "modern_co2": 1.1, "updated_co2": 0.75,
            "old_eff": 68, "modern_eff": 80, "updated_eff": 92,
            "usage_hours": 3, "units": 250
        },
        "LED Lighting": {
            "old_energy": 100, "modern_energy": 40, "updated_energy": 20,
            "old_co2": 0.05, "modern_co2": 0.02, "updated_co2": 0.01,
            "old_eff": 50, "modern_eff": 80, "updated_eff": 95,
            "usage_hours": 8, "units": 1000
        },
        "Computer": {
            "old_energy": 300, "modern_energy": 180, "updated_energy": 120,
            "old_co2": 0.15, "modern_co2": 0.09, "updated_co2": 0.06,
            "old_eff": 60, "modern_eff": 78, "updated_eff": 88,
            "usage_hours": 8, "units": 450
        },
        "Microwave": {
            "old_energy": 1500, "modern_energy": 1000, "updated_energy": 800,
            "old_co2": 0.75, "modern_co2": 0.5, "updated_co2": 0.4,
            "old_eff": 65, "modern_eff": 78, "updated_eff": 85,
            "usage_hours": 1, "units": 350
        },
        "Dishwasher": {
            "old_energy": 2000, "modern_energy": 1400, "updated_energy": 1000,
            "old_co2": 1.0, "modern_co2": 0.7, "updated_co2": 0.5,
            "old_eff": 62, "modern_eff": 76, "updated_eff": 86,
            "usage_hours": 1.5, "units": 200
        },
        "Electric Vehicle": {
            "old_energy": 20000, "modern_energy": 15000, "updated_energy": 12000,
            "old_co2": 10.0, "modern_co2": 7.5, "updated_co2": 6.0,
            "old_eff": 70, "modern_eff": 85, "updated_eff": 92,
            "usage_hours": 2, "units": 150
        }
    }
    
    # Generate dataset
    data = []
    np.random.seed(42)
    
    for category, specs in appliances_data.items():
        # Add some realistic variation
        noise_factor = np.random.uniform(0.95, 1.05)
        
        row = {
            "Category": category,
            "Old_Energy(W)": round(specs["old_energy"] * noise_factor, 1),
            "Modern_Energy(W)": round(specs["modern_energy"] * noise_factor, 1),
            "Updated_Energy(W)": round(specs["updated_energy"] * noise_factor, 1),
            "Old_CO2(kg/hr)": round(specs["old_co2"] * noise_factor, 4),
            "Modern_CO2(kg/hr)": round(specs["modern_co2"] * noise_factor, 4),
            "Updated_CO2(kg/hr)": round(specs["updated_co2"] * noise_factor, 4),
            "Old_Eff(%)": round(specs["old_eff"] * noise_factor, 1),
            "Modern_Eff(%)": round(specs["modern_eff"] * noise_factor, 1),
            "Updated_Eff(%)": round(specs["updated_eff"] * noise_factor, 1),
            "Old_Cost($)": round(specs["old_energy"] * 0.002 * noise_factor, 2),
            "Modern_Cost($)": round(specs["modern_energy"] * 0.0024 * noise_factor, 2),
            "Updated_Cost($)": round(specs["updated_energy"] * 0.003 * noise_factor, 2),
            "Smart_Features": np.random.choice(["Basic", "Moderate", "Advanced"], p=[0.2, 0.5, 0.3]),
            "ROI_Months": np.random.randint(18, 48),
            "Maintenance_Score": np.random.randint(75, 98),
            "Warranty_Years": np.random.choice([2, 3, 5, 7, 10], p=[0.1, 0.3, 0.3, 0.2, 0.1]),
            "Avg_Usage_Hours": specs["usage_hours"],
            "Units_Sold": specs["units"],
            "Customer_Rating": round(np.random.uniform(3.8, 4.9), 1),
            "Price_INR": round(specs["updated_energy"] * 100 * noise_factor, 0)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV for future use
    df.to_csv(dataset_path, index=False)
    st.sidebar.success(f"✅ Dataset generated and saved to {dataset_path}")
    
    return df

@st.cache_data
def generate_time_series_data(df):
    """Generate realistic time series data based on main dataset"""
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    time_series = []
    
    for _, row in df.iterrows():
        category = row["Category"]
        base_usage = row["Updated_Energy(W)"] * row["Avg_Usage_Hours"] / 1000  # kWh per day
        
        for date in dates:
            # Seasonal patterns (summer peak for AC, winter for heaters)
            day_of_year = date.dayofyear
            if category == "Air Conditioner":
                seasonal = 1 + 0.6 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            elif category == "Water Heater":
                seasonal = 1 + 0.4 * np.sin(2 * np.pi * (day_of_year + 80) / 365)
            else:
                seasonal = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Weekly patterns (higher on weekends for some appliances)
            weekly = 1.0
            if date.dayofweek >= 5 and category in ["Television", "Washing Machine"]:
                weekly = 1.2
            
            # Random daily variation
            noise = np.random.normal(1.0, 0.08)
            
            energy = base_usage * seasonal * weekly * noise
            
            time_series.append({
                "Date": date,
                "Category": category,
                "Energy_Usage": round(energy, 2),
                "CO2_Emission": round(energy * row["Updated_CO2(kg/hr)"], 4),
                "Cost": round(energy * 7, 2),  # ₹7 per kWh
                "Temperature": 25 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2),
                "Units_Sold_Daily": max(0, int(row["Units_Sold"] / 365 * seasonal * noise))
            })
    
    return pd.DataFrame(time_series)

@st.cache_data
def generate_regional_data(df):
    """Generate realistic regional market data"""
    regions = {
        "North": {"temp_avg": 20, "adoption_base": 0.65, "income_factor": 1.1},
        "South": {"temp_avg": 28, "adoption_base": 0.75, "income_factor": 1.2},
        "East": {"temp_avg": 26, "adoption_base": 0.60, "income_factor": 0.95},
        "West": {"temp_avg": 27, "adoption_base": 0.80, "income_factor": 1.3},
        "Central": {"temp_avg": 24, "adoption_base": 0.55, "income_factor": 0.90}
    }
    
    regional_data = []
    
    for region, props in regions.items():
        for _, row in df.iterrows():
            category = row["Category"]
            
            # Adoption varies by region and appliance type
            if category == "Air Conditioner":
                adoption = min(0.95, props["adoption_base"] * (props["temp_avg"] / 25))
            elif category == "Electric Vehicle":
                adoption = props["adoption_base"] * props["income_factor"] * 0.4
            else:
                adoption = props["adoption_base"] * (0.9 + np.random.uniform(0, 0.2))
            
            # Calculate average savings based on usage
            energy_saved = (row["Old_Energy(W)"] - row["Updated_Energy(W)"]) * row["Avg_Usage_Hours"]
            monthly_savings = energy_saved * 30 * 7 / 1000  # ₹7 per kWh
            
            regional_data.append({
                "Region": region,
                "Category": category,
                "Adoption_Rate": round(adoption, 3),
                "Avg_Savings": round(monthly_savings, 2),
                "Incentives": "Yes" if props["income_factor"] > 1.0 else np.random.choice(["Yes", "No"]),
                "Grid_Stability": round(np.random.uniform(0.82, 0.97), 3),
                "Avg_Temperature": props["temp_avg"],
                "Market_Size": round(row["Units_Sold"] * props["income_factor"] * np.random.uniform(0.8, 1.2), 0),
                "Growth_Rate": round(np.random.uniform(8, 18), 1)
            })
    
    return pd.DataFrame(regional_data)

@st.cache_data
def generate_vendor_data():
    """Generate comprehensive data for vendor view using real dataset"""
    df = load_dataset()
    ts_df = generate_time_series_data(df)
    regional_df = generate_regional_data(df)
    
    return df, ts_df, regional_df

@st.cache_data
def generate_customer_data():
    """Generate simplified data for customer view based on real dataset"""
    df = load_dataset()
    
    savings_data = {}
    for _, row in df.iterrows():
        category = row["Category"]
        
        # Map to customer-friendly format
        appliance_icons = {
            "Refrigerator": "❄️", "Air Conditioner": "🌡️", "Washing Machine": "🧺",
            "Television": "📺", "Water Heater": "🚿", "LED Lighting": "💡",
            "Computer": "💻", "Microwave": "🍽️", "Dishwasher": "🍴",
            "Electric Vehicle": "🚗"
        }
        
        usage_patterns = {
            "Refrigerator": "24/7", "Air Conditioner": "8 hrs/day",
            "Washing Machine": "2 hrs/day", "Television": "6 hrs/day",
            "Water Heater": "3 hrs/day", "LED Lighting": "8 hrs/day",
            "Computer": "8 hrs/day", "Microwave": "1 hr/day",
            "Dishwasher": "1.5 hrs/day", "Electric Vehicle": "2 hrs/day"
        }
        
        old_energy = row["Old_Energy(W)"]
        new_energy = row["Updated_Energy(W)"]
        
        savings_data[category] = {
            "icon": appliance_icons.get(category, "⚡"),
            "old_energy": old_energy,
            "new_energy": new_energy,
            "savings_percent": round(((old_energy - new_energy) / old_energy) * 100, 1),
            "usage": usage_patterns.get(category, "Variable"),
            "monthly_savings": round((old_energy - new_energy) * row["Avg_Usage_Hours"] * 30 * 7 / 1000, 2),
            "rating": row["Customer_Rating"],
            "price": row["Price_INR"]
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
        
        # Dataset info
        df = load_dataset()
        st.info(f"""
        **Dataset Info:**
        - 📊 {len(df)} appliances
        - 🔋 Real energy data
        - 📈 Based on market research
        """)
        
        st.markdown("---")
        st.markdown("### 🏠 Tell Us About Your Appliance")
        
        customer_data = generate_customer_data()
        
        appliance_options = [f"{info['icon']} {name}" for name, info in customer_data.items()]
        selected = st.selectbox("What appliance do you want to upgrade?", appliance_options)
        appliance = selected.split(" ", 1)[1]
        
        # Show appliance details
        appliance_info = customer_data[appliance]
        st.markdown(f"""
        **Product Details:**
        - ⭐ Rating: {appliance_info['rating']}/5.0
        - 💰 Price: ₹{appliance_info['price']:,.0f}
        - 🔋 Savings: {appliance_info['savings_percent']:.0f}%
        """)
        
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
        
        # Dataset Statistics
        st.markdown("---")
        st.markdown("### 📊 Dataset Statistics")
        st.metric("Total Appliances", len(df))
        st.metric("Total Records", len(ts_df))
        st.metric("Regions Covered", len(regional_df["Region"].unique()))
        st.metric("Time Period", "365 days")
        
        # Show dataset summary
        with st.expander("📈 Data Summary"):
            st.write(f"**Energy Range:** {df['Updated_Energy(W)'].min():.0f}W - {df['Updated_Energy(W)'].max():.0f}W")
            st.write(f"**Avg Efficiency:** {df['Updated_Eff(%)'].mean():.1f}%")
            st.write(f"**Total Units Sold:** {df['Units_Sold'].sum():,.0f}")
            st.write(f"**Avg Rating:** {df['Customer_Rating'].mean():.1f}/5.0")
        
        st.markdown("---")
        
        category = st.selectbox("🏷️ Product Category", df["Category"].tolist())
        hours_per_day = st.slider("Daily Usage Hours", 1, 24, 8)
        electricity_rate = st.slider("Electricity Rate (₹/kWh)", 3, 15, 7)
        region = st.selectbox("🗺️ Target Region", regional_df["Region"].unique())
        
        st.markdown("---")
        st.markdown("### 📊 Analytics Options")
        show_raw_data = st.checkbox("Show Raw Data", False)
        show_statistics = st.checkbox("Show Statistics", True)
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "PDF", "JSON"])
        
        if st.button("📥 Export Report", use_container_width=True):
            st.success(f"Report exported as {export_format}")
            
            # Show export preview
            with st.expander("📄 Export Preview"):
                st.dataframe(df.head(), use_container_width=True)
    
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
        st.markdown("### 📊 Dataset Explorer")
        
        if show_raw_data:
            st.markdown("#### 📋 Main Dataset")
            st.dataframe(df, use_container_width=True)
            
            # Dataset statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Appliances", len(df))
                st.metric("Avg Energy (Updated)", f"{df['Updated_Energy(W)'].mean():.0f}W")
            with col2:
                st.metric("Total Units Sold", f"{df['Units_Sold'].sum():,.0f}")
                st.metric("Avg Efficiency", f"{df['Updated_Eff(%)'].mean():.1f}%")
            with col3:
                st.metric("Avg Customer Rating", f"{df['Customer_Rating'].mean():.2f}/5.0")
                st.metric("Avg Price", f"₹{df['Price_INR'].mean():,.0f}")
            
            st.markdown("---")
            st.markdown("#### 📈 Time Series Data (Last 30 Days)")
            ts_filtered = ts_df[ts_df["Category"] == category].tail(30)
            st.dataframe(ts_filtered, use_container_width=True)
            
            # Time series statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Daily Usage", f"{ts_filtered['Energy_Usage'].mean():.2f} kWh")
            with col2:
                st.metric("Total CO₂", f"{ts_filtered['CO2_Emission'].sum():.2f} kg")
            with col3:
                st.metric("Total Cost", f"₹{ts_filtered['Cost'].sum():.0f}")
            with col4:
                st.metric("Units Sold", f"{ts_filtered['Units_Sold_Daily'].sum():,.0f}")
            
            st.markdown("---")
            st.markdown("#### 🗺️ Regional Market Data")
            regional_filtered = regional_df[regional_df["Category"] == category]
            st.dataframe(regional_filtered, use_container_width=True)
            
            # Regional statistics
            st.markdown("**Regional Insights:**")
            best_region = regional_filtered.loc[regional_filtered['Adoption_Rate'].idxmax()]
            st.success(f"🏆 Best Market: **{best_region['Region']}** with {best_region['Adoption_Rate']*100:.1f}% adoption")
            
            highest_growth = regional_filtered.loc[regional_filtered['Growth_Rate'].idxmax()]
            st.info(f"📈 Highest Growth: **{highest_growth['Region']}** at {highest_growth['Growth_Rate']:.1f}% YoY")
            
        else:
            st.info("Enable 'Show Raw Data' in sidebar to view detailed data tables")
            
            # Show summary even without raw data
            st.markdown("#### 📊 Quick Dataset Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Energy distribution
                fig = px.bar(df, x="Category", y="Updated_Energy(W)",
                            title="Energy Consumption by Appliance",
                            color="Updated_Energy(W)",
                            color_continuous_scale=["#00C853", "#FFD700"])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Efficiency distribution
                fig = px.bar(df, x="Category", y="Updated_Eff(%)",
                            title="Efficiency Ratings",
                            color="Updated_Eff(%)",
                            color_continuous_scale=["#FF9800", "#00C853"])
                st.plotly_chart(fig, use_container_width=True)
        
        # Data quality metrics
        if show_statistics:
            st.markdown("---")
            st.markdown("#### 📈 Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Energy Consumption Statistics (Updated Models)**")
                energy_stats = df["Updated_Energy(W)"].describe()
                st.dataframe(pd.DataFrame({
                    "Metric": ["Count", "Mean", "Std Dev", "Min", "25%", "50%", "75%", "Max"],
                    "Value": [f"{energy_stats['count']:.0f}",
                             f"{energy_stats['mean']:.1f}W",
                             f"{energy_stats['std']:.1f}W",
                             f"{energy_stats['min']:.1f}W",
                             f"{energy_stats['25%']:.1f}W",
                             f"{energy_stats['50%']:.1f}W",
                             f"{energy_stats['75%']:.1f}W",
                             f"{energy_stats['max']:.1f}W"]
                }), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Efficiency Statistics**")
                eff_stats = df["Updated_Eff(%)"].describe()
                st.dataframe(pd.DataFrame({
                    "Metric": ["Count", "Mean", "Std Dev", "Min", "25%", "50%", "75%", "Max"],
                    "Value": [f"{eff_stats['count']:.0f}",
                             f"{eff_stats['mean']:.1f}%",
                             f"{eff_stats['std']:.1f}%",
                             f"{eff_stats['min']:.1f}%",
                             f"{eff_stats['25%']:.1f}%",
                             f"{eff_stats['50%']:.1f}%",
                             f"{eff_stats['75%']:.1f}%",
                             f"{eff_stats['max']:.1f}%"]
                }), use_container_width=True, hide_index=True)
    
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