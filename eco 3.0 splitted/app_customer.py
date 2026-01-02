# ======================================================
# app_customer.py – EcoFusion 3.0 Customer Portal
# Simplified, user-friendly interface for customers
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add model directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------
# Page Config
# ------------------------------------------------------
st.set_page_config(
    page_title="🌿 EcoFusion - Your Green Energy Assistant",
    layout="wide",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# Simplified CSS - Customer Friendly
# ------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: 700;
        color: #00C853;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 2em;
    }
    
    .big-metric {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,200,83,0.2);
    }
    
    .big-metric-value {
        font-size: 3em;
        font-weight: 800;
        color: #00C853;
        margin-bottom: 10px;
    }
    
    .big-metric-label {
        font-size: 1.2em;
        color: #388E3C;
        font-weight: 600;
    }
    
    .info-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #00C853;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #FFF9C4, #FFF59D);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border-left: 5px solid #FBC02D;
    }
    
    .stButton > button {
        background: #00C853;
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1.1em;
    }
    
    .stButton > button:hover {
        background: #00E676;
        transform: scale(1.05);
    }
    
    .benefit-item {
        background: #E8F5E9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00C853;
    }
    
    h1, h2, h3 {
        color: #00C853;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# Data Generation
# ------------------------------------------------------
@st.cache_data
def generate_customer_data():
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
            "monthly_savings": (old_energy - new_energy) * 0.007 * 30  # ₹0.007 per watt
        }
    
    return savings_data

def calculate_savings(appliance, hours, rate, data):
    info = data[appliance]
    old_watts = info["old_energy"]
    new_watts = info["new_energy"]
    
    daily_old = (old_watts * hours) / 1000  # kWh
    daily_new = (new_watts * hours) / 1000  # kWh
    
    monthly_old = daily_old * 30 * rate
    monthly_new = daily_new * 30 * rate
    monthly_savings = monthly_old - monthly_new
    
    yearly_savings = monthly_savings * 12
    co2_saved = (daily_old - daily_new) * 30 * 0.5  # 0.5 kg CO2 per kWh
    
    return {
        "monthly_old": monthly_old,
        "monthly_new": monthly_new,
        "monthly_savings": monthly_savings,
        "yearly_savings": yearly_savings,
        "co2_saved": co2_saved,
        "energy_reduction": ((old_watts - new_watts) / old_watts) * 100
    }

# ------------------------------------------------------
# Header
# ------------------------------------------------------
st.markdown("<div class='main-title'>🌿 Welcome to EcoFusion</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your Personal Energy Savings Calculator</div>", unsafe_allow_html=True)

# ------------------------------------------------------
# Simple Sidebar
# ------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏠 Tell Us About Your Appliance")
    
    data = generate_customer_data()
    
    # Simple appliance selection with icons
    appliance_options = [f"{info['icon']} {name}" for name, info in data.items()]
    selected = st.selectbox("What appliance do you want to upgrade?", appliance_options)
    appliance = selected.split(" ", 1)[1]  # Remove icon
    
    st.markdown("---")
    st.markdown("### ⚡ Your Usage")
    
    hours = st.slider("How many hours per day do you use it?", 1, 24, 8,
                     help="Select average daily usage hours")
    
    rate = st.slider("What's your electricity rate? (₹/kWh)", 3, 15, 7,
                    help="Check your electricity bill for the rate per unit")
    
    st.markdown("---")
    
    # Quick info
    st.info(f"""
    **Quick Info:**
    
    {data[appliance]['icon']} Typical usage: {data[appliance]['usage']}
    
    💡 Your input: {hours} hours/day
    
    ⚡ Rate: ₹{rate}/kWh
    """)

# ------------------------------------------------------
# Calculate Savings
# ------------------------------------------------------
savings = calculate_savings(appliance, hours, rate, data)

# ------------------------------------------------------
# Hero Section - Big Numbers
# ------------------------------------------------------
st.markdown("## 💰 Your Potential Savings")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='big-metric'>
        <div class='big-metric-value'>₹{savings['monthly_savings']:.0f}</div>
        <div class='big-metric-label'>Per Month</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='big-metric'>
        <div class='big-metric-value'>₹{savings['yearly_savings']:.0f}</div>
        <div class='big-metric-label'>Per Year</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='big-metric'>
        <div class='big-metric-value'>{savings['co2_saved']:.0f} kg</div>
        <div class='big-metric-label'>CO₂ Saved Monthly</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------
# Visual Comparison
# ------------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Easy Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💡 Your Current Bill vs New Bill")
    
    # Simple bar chart
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
    
    # Pie chart
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

# ------------------------------------------------------
# Benefits Section
# ------------------------------------------------------
st.markdown("---")
st.markdown("## ✨ What This Means For You")

col1, col2 = st.columns(2)

with col1:
    trees_equivalent = savings['co2_saved'] / 21  # 21 kg CO2 per tree per year
    
    st.markdown(f"""
    <div class='benefit-item'>
        <h3>🌳 Environmental Impact</h3>
        <p><strong>You'll save {savings['co2_saved']:.0f} kg of CO₂ every month!</strong></p>
        <p>That's like planting <strong>{trees_equivalent:.1f} trees</strong> every month</p>
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
    
    payback_months = (data[appliance]['new_energy'] * 100) / savings['monthly_savings'] if savings['monthly_savings'] > 0 else 0
    
    st.markdown(f"""
    <div class='benefit-item'>
        <h3>📅 Payback Time</h3>
        <p><strong>Your investment pays back in {payback_months:.0f} months</strong></p>
        <p>After that, it's pure savings!</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------
# Recommendation
# ------------------------------------------------------
st.markdown("---")

if savings['yearly_savings'] > 5000:
    recommendation = "🚀 Highly Recommended!"
    message = "This upgrade will save you a lot of money and help the environment significantly!"
    color = "#00C853"
elif savings['yearly_savings'] > 2000:
    recommendation = "👍 Good Investment"
    message = "This upgrade makes good financial sense and reduces your carbon footprint."
    color = "#FBC02D"
else:
    recommendation = "💡 Consider It"
    message = "This upgrade will help you save energy and contribute to a greener planet."
    color = "#FF9800"

st.markdown(f"""
<div class='recommendation-box' style='border-left-color: {color};'>
    <h2>{recommendation}</h2>
    <p style='font-size: 1.2em; color: #666;'>{message}</p>
    <p style='margin-top: 20px;'><strong>Total 5-Year Savings: ₹{savings['yearly_savings'] * 5:,.0f}</strong></p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Simple Timeline
# ------------------------------------------------------
st.markdown("---")
st.markdown("## 📈 Your Savings Over Time")

months = list(range(1, 61))  # 5 years
cumulative_savings = [savings['monthly_savings'] * m for m in months]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=months,
    y=cumulative_savings,
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
    hovermode='x',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Add milestone markers
fig.add_hline(y=10000, line_dash="dash", line_color="gray", 
             annotation_text="₹10,000 saved")
fig.add_hline(y=25000, line_dash="dash", line_color="gray",
             annotation_text="₹25,000 saved")

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
# Call to Action
# ------------------------------------------------------
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🌟 Get Your Personalized Upgrade Plan", use_container_width=True):
        st.balloons()
        st.success("""
        🎉 Great choice! Here's what happens next:
        
        1. ✅ We'll send you a detailed report
        2. 📞 Our expert will contact you
        3. 🏠 Free home assessment available
        4. 💰 Get the best financing options
        
        Thank you for choosing a greener future! 🌍
        """)

# ------------------------------------------------------
# Simple Footer
# ------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 30px; color: #666;'>
    <div style='color: #00C853; font-size: 1.3em; font-weight: 600; margin-bottom: 10px;'>
        🌿 EcoFusion - Making Green Energy Easy
    </div>
    <p>Join thousands of happy customers saving money and the planet!</p>
    <p style='font-size: 0.9em; margin-top: 15px;'>
        Questions? Call us at 1800-ECO-FUSION or email support@ecofusion.com
    </p>
    <p style='font-size: 0.8em; color: #999; margin-top: 10px;'>
        © {datetime.now().year} EcoFusion | Your Partner in Sustainable Living
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Helpful Tips Section
# ------------------------------------------------------
with st.expander("💡 Quick Tips to Save Even More Energy"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Daily Habits
        - ✅ Unplug devices when not in use
        - ✅ Use natural light during the day
        - ✅ Keep your appliances clean
        - ✅ Set optimal temperatures
        """)
    
    with col2:
        st.markdown("""
        ### Smart Upgrades
        - ✅ Install LED bulbs everywhere
        - ✅ Use smart power strips
        - ✅ Consider solar panels
        - ✅ Upgrade to Energy Star appliances
        """)

# Session state for interactivity
if 'calculation_count' not in st.session_state:
    st.session_state.calculation_count = 0
st.session_state.calculation_count += 1