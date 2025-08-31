import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="HydroTwin - AI Monitoring Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1e3c72;
        margin-bottom: 1rem;
    }
    .status-normal { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-critical { color: #dc3545; }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_history' not in st.session_state:
    st.session_state.data_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Data generation functions
def generate_sensor_data():
    """Generate realistic sensor data for hydrogen plant"""
    base_time = datetime.now()
    
    # Base values with some correlation and realistic ranges
    room_temp = np.random.normal(25, 2)  # 23-27°C
    stack_current = np.random.normal(150, 10)  # 130-170A
    
    # Voltage inversely correlated with current (Ohm's law effect)
    stack_voltage = np.random.normal(48 - (stack_current-150)*0.05, 1.5)
    
    # Power = V * I / 1000 (kWh conversion factor)
    power_consumption = (stack_voltage * stack_current / 1000) + np.random.normal(0, 0.5)
    
    # Temperatures affected by current and power
    h2_temp = np.random.normal(65 + (stack_current-150)*0.1, 3)
    o2_temp = np.random.normal(60 + (stack_current-150)*0.08, 2.5)
    electrolyte_temp = np.random.normal(55 + power_consumption*0.5, 2)
    
    # Electrolyte properties
    electrolyte_concentration = np.random.normal(30, 1.5)  # 28-32%
    electrolyte_flow = np.random.normal(2.5, 0.2)  # 2.1-2.9 m³/hr
    
    return {
        'timestamp': base_time,
        'room_temperature': max(20, min(30, room_temp)),
        'stack_current': max(100, min(200, stack_current)),
        'stack_voltage': max(40, min(55, stack_voltage)),
        'power_consumption': max(4, min(12, power_consumption)),
        'h2_outlet_temp': max(50, min(80, h2_temp)),
        'o2_outlet_temp': max(45, min(75, o2_temp)),
        'electrolyte_supply_temp': max(45, min(70, electrolyte_temp)),
        'electrolyte_concentration': max(25, min(35, electrolyte_concentration)),
        'electrolyte_flow': max(2.0, min(3.0, electrolyte_flow))
    }

def train_prediction_model(data_history):
    """Train a Random Forest model for voltage prediction"""
    if len(data_history) < 10:
        return None, None
    
    df = pd.DataFrame(data_history)
    
    # Features for prediction
    features = ['room_temperature', 'stack_current', 'power_consumption', 
               'h2_outlet_temp', 'o2_outlet_temp', 'electrolyte_supply_temp',
               'electrolyte_concentration', 'electrolyte_flow']
    
    X = df[features]
    y = df['stack_voltage']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def predict_voltage(model, scaler, current_data):
    """Predict voltage for next 5-10 minutes"""
    if model is None or scaler is None:
        return None
    
    features = ['room_temperature', 'stack_current', 'power_consumption', 
               'h2_outlet_temp', 'o2_outlet_temp', 'electrolyte_supply_temp',
               'electrolyte_concentration', 'electrolyte_flow']
    
    X = np.array([[current_data[f] for f in features]])
    X_scaled = scaler.transform(X)
    
    # Add some noise to simulate future uncertainty
    base_prediction = model.predict(X_scaled)[0]
    predictions = []
    
    for i in range(5, 11):  # 5-10 minutes ahead
        noise = np.random.normal(0, 0.1 * i)  # Increasing uncertainty
        predictions.append({
            'minutes_ahead': i,
            'predicted_voltage': base_prediction + noise
        })
    
    return predictions

def calculate_rul(current_data):
    """Calculate Remaining Useful Life based on operating conditions"""
    # Simple RUL estimation based on operating stress
    voltage_stress = max(0, (50 - current_data['stack_voltage']) / 10)
    current_stress = max(0, (current_data['stack_current'] - 150) / 50)
    temp_stress = max(0, (current_data['h2_outlet_temp'] - 65) / 15)
    
    total_stress = (voltage_stress + current_stress + temp_stress) / 3
    base_rul = 8760  # 1 year in hours
    rul_hours = base_rul * (1 - total_stress * 0.3)
    
    return max(100, rul_hours)

def get_system_status(current_data, rul_hours):
    """Determine system status based on current conditions"""
    if current_data['stack_voltage'] < 45 or rul_hours < 1000:
        return "critical", "🔴"
    elif current_data['stack_voltage'] < 47 or rul_hours < 3000:
        return "warning", "🟡"
    else:
        return "normal", "🟢"

# Main Dashboard Header
st.markdown("""
<div class="main-header">
    <h1>⚡ HydroTwin - Green Hydrogen Plant Monitoring</h1>
    <p>AI-Powered Predictive Maintenance Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("🔧 Control Panel")
    
    auto_refresh = st.checkbox("Auto Refresh Data", value=True)
    refresh_interval = st.selectbox("Refresh Interval", [1, 3, 5, 10], index=1)
    
    st.divider()
    st.subheader("📊 Data Controls")
    if st.button("Reset Data History"):
        st.session_state.data_history = []
        st.session_state.model_trained = False
        st.rerun()

# Generate new data point
current_data = generate_sensor_data()
st.session_state.data_history.append(current_data)

# Keep only last 100 data points
if len(st.session_state.data_history) > 100:
    st.session_state.data_history = st.session_state.data_history[-100:]

# Train model if we have enough data
if len(st.session_state.data_history) >= 10 and not st.session_state.model_trained:
    model, scaler = train_prediction_model(st.session_state.data_history)
    st.session_state.rf_model = model
    st.session_state.scaler = scaler
    st.session_state.model_trained = True

# Main dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-time Monitoring", "🤖 AI Predictions", "⚙️ What-if Scenarios", "📈 Analytics"])

with tab1:
    st.header("Real-time Sensor Monitoring")
    
    # KPI Cards Row 1
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Room Temperature", f"{current_data['room_temperature']:.1f}°C", 
                 delta=f"{np.random.uniform(-0.5, 0.5):.1f}")
    
    with col2:
        st.metric("Stack Current", f"{current_data['stack_current']:.0f}A", 
                 delta=f"{np.random.uniform(-2, 2):.0f}")
    
    with col3:
        st.metric("Stack Voltage", f"{current_data['stack_voltage']:.1f}V", 
                 delta=f"{np.random.uniform(-0.3, 0.3):.1f}")
    
    with col4:
        st.metric("Power Consumption", f"{current_data['power_consumption']:.1f}kWh", 
                 delta=f"{np.random.uniform(-0.2, 0.2):.1f}")

    # KPI Cards Row 2
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("H₂ Outlet Temp", f"{current_data['h2_outlet_temp']:.1f}°C")
    
    with col6:
        st.metric("O₂ Outlet Temp", f"{current_data['o2_outlet_temp']:.1f}°C")
    
    with col7:
        st.metric("Electrolyte Temp", f"{current_data['electrolyte_supply_temp']:.1f}°C")
    
    with col8:
        st.metric("Electrolyte Conc.", f"{current_data['electrolyte_concentration']:.1f}%")

    # Time series charts
    if len(st.session_state.data_history) > 1:
        df_history = pd.DataFrame(st.session_state.data_history)
        
        # Voltage and Current trends
        col_left, col_right = st.columns(2)
        
        with col_left:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_history['timestamp'], 
                y=df_history['stack_voltage'],
                name='Stack Voltage (V)',
                line=dict(color='#1f77b4', width=2)
            ))
            fig1.update_layout(
                title="Stack Voltage Trend",
                xaxis_title="Time",
                yaxis_title="Voltage (V)",
                height=300
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_right:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_history['timestamp'], 
                y=df_history['stack_current'],
                name='Stack Current (A)',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig2.update_layout(
                title="Stack Current Trend",
                xaxis_title="Time",
                yaxis_title="Current (A)",
                height=300
            )
            st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("🤖 AI-Powered Predictions")
    
    if st.session_state.model_trained:
        # RUL Calculation
        rul_hours = calculate_rul(current_data)
        status, status_icon = get_system_status(current_data, rul_hours)
        
        # Status overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <h3>System Status {status_icon}</h3>
                <p class="metric-value status-{status}">{status.upper()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            rul_days = rul_hours / 24
            st.markdown(f"""
            <div class="kpi-card">
                <h3>Remaining Useful Life</h3>
                <p class="metric-value">{rul_days:.0f} days</p>
                <p class="metric-label">{rul_hours:.0f} hours</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            efficiency = 35 + np.random.uniform(-2, 2)  # kWh/kg H2
            st.markdown(f"""
            <div class="kpi-card">
                <h3>Current Efficiency</h3>
                <p class="metric-value">{efficiency:.1f}</p>
                <p class="metric-label">kWh/kg H₂</p>
            </div>
            """, unsafe_allow_html=True)
        
        # RUL Progress Bar
        st.subheader("Remaining Useful Life Progress")
        rul_percentage = min(100, (rul_hours / 8760) * 100)
        
        if rul_percentage > 50:
            bar_color = "🟢"
        elif rul_percentage > 20:
            bar_color = "🟡"
        else:
            bar_color = "🔴"
            
        st.progress(rul_percentage / 100)
        st.write(f"{bar_color} {rul_percentage:.1f}% of expected lifetime remaining")
        
        # Voltage Predictions
        st.subheader("Voltage Prediction (Next 10 minutes)")
        predictions = predict_voltage(st.session_state.rf_model, st.session_state.scaler, current_data)
        
        if predictions:
            pred_df = pd.DataFrame(predictions)
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=[0], 
                y=[current_data['stack_voltage']],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Current Voltage'
            ))
            fig_pred.add_trace(go.Scatter(
                x=pred_df['minutes_ahead'], 
                y=pred_df['predicted_voltage'],
                mode='lines+markers',
                name='Predicted Voltage',
                line=dict(color='blue', dash='dash')
            ))
            fig_pred.update_layout(
                title="Voltage Prediction",
                xaxis_title="Minutes Ahead",
                yaxis_title="Voltage (V)",
                height=400
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        
        # Anomaly Detection
        st.subheader("Anomaly Detection")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp_anomaly = "Normal" if current_data['h2_outlet_temp'] < 75 else "High"
            temp_color = "🟢" if temp_anomaly == "Normal" else "🟡"
            st.write(f"{temp_color} **H₂ Temperature**: {temp_anomaly}")
        
        with col2:
            voltage_anomaly = "Normal" if current_data['stack_voltage'] > 45 else "Low"
            voltage_color = "🟢" if voltage_anomaly == "Normal" else "🔴"
            st.write(f"{voltage_color} **Stack Voltage**: {voltage_anomaly}")
        
        with col3:
            flow_anomaly = "Normal" if 2.0 < current_data['electrolyte_flow'] < 3.0 else "Abnormal"
            flow_color = "🟢" if flow_anomaly == "Normal" else "🟡"
            st.write(f"{flow_color} **Electrolyte Flow**: {flow_anomaly}")
    
    else:
        st.info("🤖 Collecting data to train AI model... Please wait for more data points.")
        st.progress(len(st.session_state.data_history) / 10)

with tab3:
    st.header("⚙️ What-if Scenario Analysis")
    
    st.write("Simulate different operating conditions and see their impact:")
    
    # Input controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Operating Conditions")
        current_change = st.slider("Current Load Change (%)", -20, 20, 0)
        concentration_change = st.slider("Electrolyte Concentration Change (%)", -10, 10, 0)
        cooling_efficiency = st.slider("Cooling System Efficiency (%)", 70, 110, 100)
    
    # Calculate scenario impacts
    scenario_current = current_data['stack_current'] * (1 + current_change/100)
    scenario_concentration = current_data['electrolyte_concentration'] * (1 + concentration_change/100)
    scenario_cooling_factor = cooling_efficiency / 100
    
    # Impact calculations
    base_efficiency = 35.0
    efficiency_impact = -abs(current_change) * 0.1 + (concentration_change * 0.05)
    scenario_efficiency = base_efficiency + efficiency_impact
    
    # Safety risk (crossover probability)
    base_safety_risk = 5.0  # Base 5% risk
    safety_impact = abs(current_change) * 0.2 + abs(concentration_change) * 0.3
    scenario_safety_risk = min(25, base_safety_risk + safety_impact)
    
    # Lifetime impact
    base_lifetime = 8760  # hours
    lifetime_reduction = (abs(current_change) + abs(concentration_change) + abs(cooling_efficiency-100)) * 10
    scenario_lifetime = max(1000, base_lifetime - lifetime_reduction)
    
    with col2:
        st.subheader("Predicted Impacts")
        
        st.metric("Efficiency", f"{scenario_efficiency:.1f} kWh/kg H₂", 
                 delta=f"{efficiency_impact:.1f}")
        
        risk_color = "🟢" if scenario_safety_risk < 10 else "🟡" if scenario_safety_risk < 20 else "🔴"
        st.metric("Safety Risk", f"{scenario_safety_risk:.1f}%", 
                 delta=f"{scenario_safety_risk - base_safety_risk:.1f}")
        
        lifetime_days = scenario_lifetime / 24
        st.metric("Expected Lifetime", f"{lifetime_days:.0f} days", 
                 delta=f"{(scenario_lifetime - base_lifetime)/24:.0f}")
    
    with col3:
        st.subheader("Scenario Summary")
        
        # Risk assessment
        if scenario_safety_risk < 10 and efficiency_impact > -2:
            st.success("✅ Optimal operating conditions")
        elif scenario_safety_risk < 20:
            st.warning("⚠️ Acceptable with monitoring")
        else:
            st.error("❌ High risk - not recommended")
        
        # Recommendations
        st.write("**Recommendations:**")
        if current_change > 10:
            st.write("• Reduce current load to improve lifetime")
        if abs(concentration_change) > 5:
            st.write("• Monitor electrolyte concentration closely")
        if cooling_efficiency < 90:
            st.write("• Improve cooling system performance")
        if scenario_safety_risk > 15:
            st.write("• ⚠️ High crossover risk detected")

    # Scenario comparison chart
    st.subheader("Scenario Comparison")
    
    scenarios = pd.DataFrame({
        'Scenario': ['Current', 'Optimized', 'High Load', 'Conservative'],
        'Efficiency': [base_efficiency, base_efficiency + 2, base_efficiency - 3, base_efficiency + 1],
        'Safety Risk': [base_safety_risk, base_safety_risk - 2, base_safety_risk + 8, base_safety_risk - 1],
        'Lifetime (days)': [base_lifetime/24, (base_lifetime + 500)/24, (base_lifetime - 1000)/24, (base_lifetime + 200)/24]
    })
    
    fig_scenario = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Efficiency (kWh/kg H₂)', 'Safety Risk (%)', 'Lifetime (days)'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig_scenario.add_trace(go.Bar(x=scenarios['Scenario'], y=scenarios['Efficiency'], name='Efficiency'), row=1, col=1)
    fig_scenario.add_trace(go.Bar(x=scenarios['Scenario'], y=scenarios['Safety Risk'], name='Safety Risk'), row=1, col=2)
    fig_scenario.add_trace(go.Bar(x=scenarios['Scenario'], y=scenarios['Lifetime (days)'], name='Lifetime'), row=1, col=3)
    
    fig_scenario.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_scenario, use_container_width=True)

with tab4:
    st.header("📈 Historical Analytics")
    
    if len(st.session_state.data_history) > 5:
        df_analytics = pd.DataFrame(st.session_state.data_history)
        
        # Performance summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_voltage = df_analytics['stack_voltage'].mean()
            st.metric("Average Voltage", f"{avg_voltage:.1f}V")
            
            voltage_stability = df_analytics['stack_voltage'].std()
            st.metric("Voltage Stability", f"{voltage_stability:.2f}V")
        
        with col2:
            avg_current = df_analytics['stack_current'].mean()
            st.metric("Average Current", f"{avg_current:.0f}A")
            
            current_stability = df_analytics['stack_current'].std()
            st.metric("Current Stability", f"{current_stability:.1f}A")
        
        with col3:
            avg_power = df_analytics['power_consumption'].mean()
            st.metric("Average Power", f"{avg_power:.1f}kWh")
            
            avg_h2_temp = df_analytics['h2_outlet_temp'].mean()
            st.metric("Avg H₂ Temperature", f"{avg_h2_temp:.1f}°C")
        
        # Correlation matrix
        st.subheader("Parameter Correlations")
        numeric_cols = ['stack_voltage', 'stack_current', 'power_consumption', 
                       'h2_outlet_temp', 'o2_outlet_temp', 'electrolyte_supply_temp']
        corr_matrix = df_analytics[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                            color_continuous_scale='RdBu_r',
                            aspect='auto',
                            title='Parameter Correlation Matrix')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Historical trends
        st.subheader("Historical Trends")
        
        # Multi-parameter chart
        fig_multi = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Voltage vs Current', 'Temperature Trends', 'Power Consumption', 'Electrolyte Properties']
        )
        
        # Voltage vs Current
        fig_multi.add_trace(go.Scatter(
            x=df_analytics['stack_current'], 
            y=df_analytics['stack_voltage'],
            mode='markers',
            name='V vs I'
        ), row=1, col=1)
        
        # Temperature trends
        fig_multi.add_trace(go.Scatter(
            x=list(range(len(df_analytics))), 
            y=df_analytics['h2_outlet_temp'],
            name='H₂ Temp'
        ), row=1, col=2)
        fig_multi.add_trace(go.Scatter(
            x=list(range(len(df_analytics))), 
            y=df_analytics['o2_outlet_temp'],
            name='O₂ Temp'
        ), row=1, col=2)
        
        # Power consumption
        fig_multi.add_trace(go.Scatter(
            x=list(range(len(df_analytics))), 
            y=df_analytics['power_consumption'],
            name='Power'
        ), row=2, col=1)
        
        # Electrolyte properties
        fig_multi.add_trace(go.Scatter(
            x=list(range(len(df_analytics))), 
            y=df_analytics['electrolyte_concentration'],
            name='Concentration'
        ), row=2, col=2)
        fig_multi.add_trace(go.Scatter(
            x=list(range(len(df_analytics))), 
            y=df_analytics['electrolyte_flow'],
            name='Flow'
        ), row=2, col=2)
        
        fig_multi.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig_multi, use_container_width=True)
    
    else:
        st.info("📊 Collecting more data for analytics... Current data points: " + str(len(st.session_state.data_history)))

# Auto refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏆 <strong>HydroTwin Dashboard</strong> - Hackathon Project for Predictive Maintenance in Green Hydrogen Plants</p>
    <p>Built with Streamlit • Real-time AI Monitoring • Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)