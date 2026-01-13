import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# --- 1. CONFIG ---
st.set_page_config(
    page_title="Kaeser Smart-Enterprise AI", 
    page_icon="🏢", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS CUSTOM (Enhanced Design) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #002B49 0%, #005293 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(0, 43, 73, 0.15);
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #005293;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .metric-card h4 {
        color: #64748b !important;
        font-size: 0.85rem;
        margin-bottom: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        color: #1e293b !important;
        font-size: 2rem;
        font-weight: 800;
        margin: 5px 0;
    }
    
    .metric-card p {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        margin-top: 5px;
    }
    
    .risk-high { border-left-color: #ef4444 !important; }
    .risk-medium { border-left-color: #f59e0b !important; }
    .risk-low { border-left-color: #10b981 !important; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #002B49 0%, #003A66 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #005293 0%, #0080C9 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #003A66 0%, #005293 100%);
        transform: scale(1.05);
    }
    
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SESSION STATE INITIALIZATION ---
if 'fleet_data' not in st.session_state:
    # Initialize with dummy data
    st.session_state.fleet_data = None
if 'unit_counter' not in st.session_state:
    st.session_state.unit_counter = 21
if 'show_add_unit' not in st.session_state:
    st.session_state.show_add_unit = False

# --- 4. ENHANCED DUMMY DATA GENERATOR WITH EXPLICIT FORMULAS ---
@st.cache_data
def load_enterprise_data():
    n_units = 20
    
    # --- SIMULASI RUMUS HEALTH SCORE (Dijabarkan) ---
    # Health Score = 100 - (Age_Penalty + Usage_Penalty + Maintenance_Penalty)
    # Age_Penalty = (Days_Since_Last_Service / 365) * 40
    # Usage_Penalty = Random(0, 20) based on operational hours
    # Maintenance_Penalty = Random(0, 15) based on maintenance history
    
    base_scores = []
    age_penalties = []
    usage_penalties = []
    maintenance_penalties = []
    
    for i in range(n_units):
        # Random days since last service (10-200 days)
        days_since_service = np.random.randint(10, 200)
        
        # Calculate penalties (EXPLICIT FORMULA)
        age_penalty = min(40, (days_since_service / 365) * 40)
        usage_penalty = np.random.randint(0, 20)
        maintenance_penalty = np.random.randint(0, 15)
        
        # Total penalty
        total_penalty = age_penalty + usage_penalty + maintenance_penalty
        
        # Health Score (100 - total_penalty)
        health_score = max(40, 100 - total_penalty)
        
        base_scores.append(int(health_score))
        age_penalties.append(round(age_penalty, 1))
        usage_penalties.append(usage_penalty)
        maintenance_penalties.append(maintenance_penalty)
    
    # Determine Status based on Health Score
    statuses = []
    for score in base_scores:
        if score < 60:
            statuses.append('Critical')
        elif score < 85:
            statuses.append('Warning')
        else:
            statuses.append('Healthy')
    
    # Generate locations with distribution
    locations = np.random.choice(
        ['Jakarta', 'Surabaya', 'Bandung', 'Semarang', 'Medan', 'Makassar'],
        n_units,
        p=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
    )
    
    # Generate coordinates based on locations
    lat_map = {
        'Jakarta': -6.2, 'Surabaya': -7.25, 'Bandung': -6.9,
        'Semarang': -6.99, 'Medan': 3.59, 'Makassar': -5.13
    }
    lon_map = {
        'Jakarta': 106.8, 'Surabaya': 112.75, 'Bandung': 107.6,
        'Semarang': 110.42, 'Medan': 98.67, 'Makassar': 119.4
    }
    
    lats = [lat_map[loc] + np.random.uniform(-0.1, 0.1) for loc in locations]
    lons = [lon_map[loc] + np.random.uniform(-0.1, 0.1) for loc in locations]
    
    # Operational hours per day (affects energy cost)
    operational_hours = np.random.uniform(8, 24, n_units).round(1)
    
    # Power consumption in kW
    power_consumption = np.random.uniform(50, 150, n_units).round(1)
    
    # Daily energy cost = operational_hours * power_consumption * electricity_rate
    electricity_rate = 1500  # Rp/kWh
    daily_energy_cost = (operational_hours * power_consumption * electricity_rate).round(0)
    
    fleet_data = pd.DataFrame({
        'Unit_ID': [f'K-DX-{i:03}' for i in range(1, n_units + 1)],
        'Lokasi': locations,
        'Status': statuses,
        'Health_Score': base_scores,
        'Age_Penalty': age_penalties,
        'Usage_Penalty': usage_penalties,
        'Maintenance_Penalty': maintenance_penalties,
        'Last_Service': [datetime.now() - timedelta(days=int(np.random.randint(10, 200))) for _ in range(n_units)],
        'Next_Service_Due': [datetime.now() + timedelta(days=int(np.random.randint(30, 180))) for _ in range(n_units)],
        'Operational_Hours_Daily': operational_hours,
        'Power_Consumption_kW': power_consumption,
        'Daily_Energy_Cost_Rp': daily_energy_cost,
        'Latitude': lats,
        'Longitude': lons,
        'Installation_Date': [datetime.now() - timedelta(days=int(np.random.randint(365, 1825))) for _ in range(n_units)]
    })
    
    return fleet_data

# Load initial data
fleet = load_enterprise_data()

# --- 5. SIDEBAR WITH UNIT MANAGEMENT ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Kaeser_Kompressoren_Logo.svg/1280px-Kaeser_Kompressoren_Logo.svg.png", 
             width=200, use_container_width=True)
    
    st.markdown("---")
    
    # Navigation Menu
    menu = st.radio("**Enterprise Navigation**", [
        "📊 Executive Dashboard",
        "🌐 Fleet Management Control",
        "🧠 AI Diagnostic Laboratory",
        "📅 Smart Maintenance Calendar",
        "⚡ Energy & ESG Sustainability",
        "💰 Financial Loss & ROI Analysis",
        "🔧 Unit Management"
    ], key='nav_menu')
    
    st.markdown("---")
    
    # Unit Management in Sidebar (for all pages)
    st.subheader("🔧 Unit Operations")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("➕ Add Unit", use_container_width=True):
            st.session_state.show_add_unit = True
    
    with col_s2:
        if st.button("🗑️ Remove Unit", use_container_width=True, type="secondary"):
            st.session_state.show_remove_unit = True
    
    # Quick Stats in Sidebar
    st.markdown("---")
    st.subheader("🚨 Live Alerts")
    
    critical_units = len(fleet[fleet['Status'] == 'Critical'])
    warning_units = len(fleet[fleet['Status'] == 'Warning'])
    
    if critical_units > 0:
        st.error(f"**Critical Units:** {critical_units}")
    if warning_units > 0:
        st.warning(f"**Warning Units:** {warning_units}")
    
    st.markdown("---")
    
    # System Info
    st.info(f"""
    **Admin Node:** Jakarta-Central
    **Total Units:** {len(fleet)}
    **System Time:** {datetime.now().strftime("%d %b %Y %H:%M:%S")}
    """)

# --- 6. UNIT MANAGEMENT PAGE ---
if menu == "🔧 Unit Management":
    st.markdown('<div class="main-header"><h1>🔧 Unit Management Console</h1><p>Add or remove units from the fleet database</p></div>', unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns([2, 1])
    
    with col_m1:
        st.subheader("Current Fleet Overview")
        st.dataframe(fleet[['Unit_ID', 'Lokasi', 'Status', 'Health_Score', 'Last_Service']].sort_values('Health_Score'), 
                    use_container_width=True, height=400)
    
    with col_m2:
        st.subheader("Quick Actions")
        
        # Add Unit Form
        with st.expander("➕ Add New Unit", expanded=st.session_state.get('show_add_unit', False)):
            with st.form("add_unit_form"):
                new_unit_id = st.text_input("Unit ID", value=f"K-DX-{st.session_state.unit_counter:03}")
                new_location = st.selectbox("Location", ['Jakarta', 'Surabaya', 'Bandung', 'Semarang', 'Medan', 'Makassar'])
                new_health_score = st.slider("Initial Health Score", 40, 100, 85)
                
                if st.form_submit_button("Add Unit to Fleet"):
                    # Create new unit entry
                    new_unit = pd.DataFrame({
                        'Unit_ID': [new_unit_id],
                        'Lokasi': [new_location],
                        'Status': ['Healthy' if new_health_score >= 85 else 'Warning' if new_health_score >= 60 else 'Critical'],
                        'Health_Score': [new_health_score],
                        'Age_Penalty': [0],
                        'Usage_Penalty': [0],
                        'Maintenance_Penalty': [0],
                        'Last_Service': [datetime.now()],
                        'Next_Service_Due': [datetime.now() + timedelta(days=90)],
                        'Operational_Hours_Daily': [np.random.uniform(8, 16).round(1)],
                        'Power_Consumption_kW': [np.random.uniform(50, 150).round(1)],
                        'Daily_Energy_Cost_Rp': [0],
                        'Latitude': [-6.2 + np.random.uniform(-0.1, 0.1)],
                        'Longitude': [106.8 + np.random.uniform(-0.1, 0.1)],
                        'Installation_Date': [datetime.now()]
                    })
                    
                    # Calculate energy cost
                    new_unit['Daily_Energy_Cost_Rp'] = (new_unit['Operational_Hours_Daily'] * 
                                                       new_unit['Power_Consumption_kW'] * 1500).round(0)
                    
                    # Append to fleet
                    fleet = pd.concat([fleet, new_unit], ignore_index=True)
                    st.session_state.unit_counter += 1
                    st.session_state.show_add_unit = False
                    st.success(f"Unit {new_unit_id} successfully added to fleet!")
                    st.rerun()
        
        # Remove Unit Form
        with st.expander("🗑️ Remove Unit", expanded=st.session_state.get('show_remove_unit', False)):
            unit_to_remove = st.selectbox("Select Unit to Remove", fleet['Unit_ID'].tolist())
            
            if st.button("Confirm Removal", type="primary"):
                fleet = fleet[fleet['Unit_ID'] != unit_to_remove]
                st.session_state.show_remove_unit = False
                st.error(f"Unit {unit_to_remove} has been removed from fleet!")
                st.rerun()
        
        # Fleet Statistics
        st.metric("Total Units", len(fleet))
        st.metric("Average Health Score", f"{fleet['Health_Score'].mean():.1f}/100")

# --- 7. EXECUTIVE DASHBOARD (NEW) ---
elif menu == "📊 Executive Dashboard":
    st.markdown('<div class="main-header"><h1>📊 Executive Dashboard</h1><p>Real-time overview of enterprise operations and KPIs</p></div>', unsafe_allow_html=True)
    
    # Top Level KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h4>🟢 System Reliability</h4>
            <h2>99.92%</h2>
            <p>vs Target: 98.5% | +1.42%</p>
            <small>Formula: (Total Uptime Hours / Total Hours) × 100%</small>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        avg_health = fleet['Health_Score'].mean()
        st.markdown(f'''
        <div class="metric-card">
            <h4>⚕️ Avg Health Score</h4>
            <h2>{avg_health:.1f}/100</h2>
            <p>Stable Trend | Last 30d: +2.3%</p>
            <small>Formula: 100 - Σ(Age + Usage + Maintenance Penalties)</small>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        total_energy_cost = fleet['Daily_Energy_Cost_Rp'].sum() * 30
        st.markdown(f'''
        <div class="metric-card">
            <h4>💰 Monthly Energy Cost</h4>
            <h2>Rp {total_energy_cost:,.0f}</h2>
            <p>Efficiency: 12.4% vs Last Month</p>
            <small>Formula: Σ(Hours × Power × Electricity Rate × 30)</small>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        online_units = len(fleet[fleet['Health_Score'] > 50])
        st.markdown(f'''
        <div class="metric-card">
            <h4>🔧 Units Online</h4>
            <h2>{online_units}/{len(fleet)}</h2>
            <p>{len(fleet) - online_units} Units Offline</p>
            <small>Operational Status: Health Score > 50</small>
        </div>
        ''', unsafe_allow_html=True)
    
    # Charts Row
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Health Score Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(fleet['Health_Score'], bins=20, color='#005293', edgecolor='black')
        ax.set_xlabel('Health Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Health Scores')
        st.pyplot(fig)
    
    with col_chart2:
        st.subheader("Status by Location")
        status_by_loc = fleet.groupby(['Lokasi', 'Status']).size().unstack().fillna(0)
        st.bar_chart(status_by_loc, height=300)
    
    # Recent Alerts
    st.markdown("---")
    st.subheader("🚨 Recent Alerts & Notifications")
    
    critical_df = fleet[fleet['Status'] == 'Critical'].sort_values('Health_Score').head(5)
    if not critical_df.empty:
        for _, row in critical_df.iterrows():
            with st.expander(f"🔴 CRITICAL: {row['Unit_ID']} - Health Score: {row['Health_Score']}", expanded=True):
                st.write(f"**Location:** {row['Lokasi']}")
                st.write(f"**Breakdown:** Age Penalty: {row['Age_Penalty']} | Usage Penalty: {row['Usage_Penalty']} | Maintenance Penalty: {row['Maintenance_Penalty']}")
                st.progress(row['Health_Score']/100)

# --- 8. FLEET MANAGEMENT CONTROL (Updated) ---
elif menu == "🌐 Fleet Management Control":
    st.markdown('<div class="main-header"><h1>🌐 Global Fleet Control Center</h1><p>Real-time monitoring and geographical distribution of all units</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Real-time Unit Distribution")
        
        # Enhanced map with status colors
        fleet['color'] = fleet['Status'].map({'Healthy': '#10b981', 'Warning': '#f59e0b', 'Critical': '#ef4444'})
        st.map(fleet, latitude='Latitude', longitude='Longitude', color='color')
    
    with col2:
        st.subheader("Regional Quick Stats")
        selected_loc = st.selectbox("Filter Region", ["All Regions"] + list(fleet['Lokasi'].unique()))
        
        if selected_loc == "All Regions":
            df_filtered = fleet
        else:
            df_filtered = fleet[fleet['Lokasi'] == selected_loc]
        
        st.write(f"**Showing {len(df_filtered)} units in {selected_loc}**")
        
        # Metrics
        healthy = len(df_filtered[df_filtered['Status'] == 'Healthy'])
        warning = len(df_filtered[df_filtered['Status'] == 'Warning'])
        critical = len(df_filtered[df_filtered['Status'] == 'Critical'])
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Healthy", healthy)
        with col_m2:
            st.metric("Warning", warning)
        with col_m3:
            st.metric("Critical", critical, delta_color="inverse")
        
        st.markdown("---")
        
        # Unit Details
        st.subheader("Unit Details")
        for i, row in df_filtered.sort_values('Health_Score').iterrows():
            status_icon = "🟢" if row['Status'] == 'Healthy' else "🟡" if row['Status'] == 'Warning' else "🔴"
            with st.expander(f"{status_icon} {row['Unit_ID']} - {row['Status']}"):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write(f"**Health Score:** {row['Health_Score']}/100")
                    st.progress(row['Health_Score']/100)
                    st.write(f"**Location:** {row['Lokasi']}")
                with col_d2:
                    st.write(f"**Last Service:** {row['Last_Service'].strftime('%d %b %Y')}")
                    st.write(f"**Next Service:** {row['Next_Service_Due'].strftime('%d %b %Y')}")
                
                # Penalty Breakdown
                st.caption(f"**Penalty Breakdown:** Age: {row['Age_Penalty']} | Usage: {row['Usage_Penalty']} | Maintenance: {row['Maintenance_Penalty']}")

# --- 9. AI DIAGNOSTIC LABORATORY (Enhanced with Risk Explanation) ---
elif menu == "🧠 AI Diagnostic Laboratory":
    st.markdown('<div class="main-header"><h1>🧠 AI Diagnostic Laboratory</h1><p>Advanced anomaly detection with machine learning algorithms</p></div>', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        loc_f = st.selectbox("Filter Region", fleet['Lokasi'].unique())
    
    with col_f2:
        unit_f = st.selectbox("Select Unit ID", fleet[fleet['Lokasi'] == loc_f]['Unit_ID'])
    
    with col_f3:
        sens = st.slider("AI Sensitivity Threshold", 0.01, 0.20, 0.10, 0.01,
                        help="Lower values detect only severe anomalies. Higher values detect more subtle anomalies.")
    
    with col_f4:
        st.markdown("### Risk Interpretation")
        risk_percent = int(sens * 100)
        if sens <= 0.05:
            st.success(f"**{risk_percent}%**: Low Sensitivity - Only critical issues detected")
        elif sens <= 0.10:
            st.info(f"**{risk_percent}%**: Moderate Sensitivity - Balanced detection")
        elif sens <= 0.15:
            st.warning(f"**{risk_percent}%**: High Sensitivity - Detects minor anomalies")
        else:
            st.error(f"**{risk_percent}%**: Very High Sensitivity - May include false positives")
    
    st.markdown("---")
    
    # ML Simulation with Enhanced Data
    n = 500
    # Generate normal data
    normal_temp = np.random.normal(70, 2, n)
    normal_pressure = np.random.normal(7, 0.3, n)
    
    # Generate anomalies based on sensitivity
    n_anomalies = int(n * sens * 2)
    anomaly_temp = np.random.normal(90, 8, n_anomalies)
    anomaly_pressure = np.random.normal(4, 1.5, n_anomalies)
    
    df_diag = pd.DataFrame({
        'Suhu': np.concatenate([normal_temp, anomaly_temp]),
        'Tekanan': np.concatenate([normal_pressure, anomaly_pressure])
    })
    
    # Apply Isolation Forest
    model = IsolationForest(contamination=sens, random_state=42)
    df_diag['Prediksi'] = model.fit_predict(df_diag[['Suhu', 'Tekanan']])
    df_diag['Label'] = np.where(df_diag['Prediksi'] == -1, 'Anomali', 'Normal')
    
    col_c1, col_c2 = st.columns([2, 1])
    
    with col_c1:
        st.subheader(f"🧪 Sensor Pattern Analysis: {unit_f}")
        
        # Create matplotlib scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot normal points
        normal_data = df_diag[df_diag['Label'] == 'Normal']
        ax.scatter(normal_data['Suhu'], normal_data['Tekanan'], 
                  alpha=0.6, s=50, color='#005293', label='Normal')
        
        # Plot anomaly points
        anomaly_data = df_diag[df_diag['Label'] == 'Anomali']
        ax.scatter(anomaly_data['Suhu'], anomaly_data['Tekanan'], 
                  alpha=0.8, s=60, color='#ef4444', label='Anomali')
        
        # Add ideal operating zone rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((65, 6.5), 10, 1, 
                        linewidth=2, linestyle='--', 
                        edgecolor='green', facecolor='green', alpha=0.1)
        ax.add_patch(rect)
        ax.text(68, 6.3, 'Ideal Zone', color='green', fontsize=10)
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Pressure (Bar)')
        ax.set_title(f'Temperature vs Pressure Distribution (Sensitivity: {sens})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col_c2:
        st.subheader("🔍 AI Diagnostic Insights")
        
        # Calculate statistics
        n_anomalies = len(df_diag[df_diag['Prediksi'] == -1])
        anomaly_percent = (n_anomalies / len(df_diag)) * 100
        
        st.metric("Detected Anomalies", n_anomalies, f"{anomaly_percent:.1f}%")
        
        # Generate insight based on anomalies
        if anomaly_percent > 15:
            st.error("""
            ## 🔴 KRITIS: HIGH RISK DETECTED
            
            **Indikasi Kebocoran Aktif:**
            - Tekanan drop 40-60% dari normal
            - Suhu motor meningkat 20-30°C
            - Efisiensi energi turun 35%
            
            **Rekomendasi:**
            1. Immediate shutdown unit
            2. Check valve seals & piping
            3. Replace air filters
            4. Schedule emergency maintenance
            """)
            
            # Calculate potential loss
            unit_power = fleet.loc[fleet['Unit_ID'] == unit_f, 'Power_Consumption_kW'].values[0]
            hourly_loss = unit_power * 1500 * 0.35  # 35% efficiency loss
            st.warning(f"**Potensi Kerugian:** Rp {hourly_loss:,.0f}/jam")
            
        elif anomaly_percent > 5:
            st.warning("""
            ## 🟡 WARNING: MEDIUM RISK
            
            **Indikasi Degradasi:**
            - Tekanan turun 10-20%
            - Suhu sedikit meningkat
            - Efisiensi turun 10-15%
            
            **Rekomendasi:**
            1. Monitor intensif 24/7
            2. Schedule preventive maintenance
            3. Check filter condition
            """)
        else:
            st.success("""
            ## 🟢 NORMAL: LOW RISK
            
            **Status Operasional:**
            - Parameter dalam batas normal
            - Kompresi optimal
            - Efisiensi energi maksimal
            
            **Maintenance:**
            - Continue routine checks
            - Next service as scheduled
            """)
        
        # Show ML parameters
        with st.expander("📊 ML Model Parameters"):
            st.write(f"**Algorithm:** Isolation Forest")
            st.write(f"**Contamination:** {sens}")
            st.write(f"**Samples:** {len(df_diag)}")
            st.write(f"**Features:** Temperature, Pressure")
            st.write(f"**Detection Rate:** {anomaly_percent:.1f}%")

# --- 10. SMART MAINTENANCE CALENDAR (Enhanced) ---
elif menu == "📅 Smart Maintenance Calendar":
    st.markdown('<div class="main-header"><h1>📅 Predictive Maintenance Planner</h1><p>AI-powered failure prediction and maintenance scheduling</p></div>', unsafe_allow_html=True)
    
    # Generate predictive maintenance schedule
    today = datetime.now()
    
    maintenance_schedule = pd.DataFrame({
        'Unit_ID': fleet['Unit_ID'].sample(8, random_state=42),
        'Location': fleet.loc[fleet['Unit_ID'].isin(fleet['Unit_ID'].sample(8, random_state=42)), 'Lokasi'].values,
        'Component': ['Air Filter', 'Oil Separator', 'Motor Bearing', 'Coupling', 
                     'Cooling System', 'Pressure Valve', 'Control Board', 'Compressor Unit'],
        'Predicted_Failure': [today + timedelta(days=x) for x in [5, 15, 30, 45, 60, 90, 120, 180]],
        'Health_Score': fleet.loc[fleet['Unit_ID'].isin(fleet['Unit_ID'].sample(8, random_state=42)), 'Health_Score'].values,
        'Risk_Level': ['High', 'Critical', 'High', 'Medium', 'Low', 'Medium', 'High', 'Low'],
        'Estimated_Cost_Rp': [2500000, 8500000, 12500000, 4500000, 3200000, 6800000, 9500000, 18500000]
    })
    
    # Sort by predicted failure date
    maintenance_schedule = maintenance_schedule.sort_values('Predicted_Failure')
    
    col_mt1, col_mt2 = st.columns([3, 1])
    
    with col_mt1:
        st.subheader("📋 Maintenance Schedule")
        
        # Display as interactive table
        st.dataframe(maintenance_schedule.style.apply(
            lambda x: ['background-color: #fef2f2' if x['Risk_Level'] == 'Critical' else 
                      'background-color: #fffbeb' if x['Risk_Level'] == 'High' else 
                      'background-color: #f0fdf4' for _ in x],
            axis=1
        ), use_container_width=True, height=400)
    
    with col_mt2:
        st.subheader("🔍 Risk Analysis")
        
        selected_unit = st.selectbox("Select Unit for Analysis", maintenance_schedule['Unit_ID'].unique())
        
        unit_data = maintenance_schedule[maintenance_schedule['Unit_ID'] == selected_unit].iloc[0]
        
        # Risk Level Display
        risk_color = {
            'Critical': '#ef4444',
            'High': '#f59e0b',
            'Medium': '#3b82f6',
            'Low': '#10b981'
        }
        
        st.markdown(f"""
        <div style="background-color: {risk_color[unit_data['Risk_Level']]}20; 
                    padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color[unit_data['Risk_Level']]};">
            <h3 style="color: {risk_color[unit_data['Risk_Level']]}; margin-top: 0;">
                {unit_data['Risk_Level']} RISK
            </h3>
            <p><strong>Unit:</strong> {unit_data['Unit_ID']}</p>
            <p><strong>Component:</strong> {unit_data['Component']}</p>
            <p><strong>Predicted Failure:</strong> {unit_data['Predicted_Failure'].strftime('%d %b %Y')}</p>
            <p><strong>Health Score:</strong> {unit_data['Health_Score']}/100</p>
            <p><strong>Est. Cost:</strong> Rp {unit_data['Estimated_Cost_Rp']:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Impact Calculation
        days_to_failure = (unit_data['Predicted_Failure'] - today).days
        
        if unit_data['Risk_Level'] == 'Critical':
            st.error(f"""
            ⚠️ **URGENT ACTION REQUIRED**
            
            **If {unit_data['Component']} fails on {unit_data['Predicted_Failure'].strftime('%d %b %Y')}:**
            - Downtime: 48-72 hours
            - Production Loss: Rp 120-180 juta
            - Emergency Repair Cost: +40%
            - Total Potential Loss: **Rp {unit_data['Estimated_Cost_Rp'] * 1.4:,.0f}**
            
            **Recommendation:** Schedule maintenance within {max(1, days_to_failure - 7)} days
            """)
        elif unit_data['Risk_Level'] == 'High':
            st.warning(f"""
            ⚠️ **HIGH PRIORITY**
            
            **Potential Impact:**
            - Downtime: 24-48 hours
            - Production Loss: Rp 60-90 juta
            - Repair Cost: Rp {unit_data['Estimated_Cost_Rp']:,.0f}
            
            **Recommendation:** Schedule within {max(1, days_to_failure - 14)} days
            """)
        else:
            st.info(f"""
            ✅ **PLANNED MAINTENANCE**
            
            **Schedule:** {unit_data['Predicted_Failure'].strftime('%d %b %Y')}
            **Estimated Cost:** Rp {unit_data['Estimated_Cost_Rp']:,.0f}
            **Days Remaining:** {days_to_failure} days
            
            **Recommendation:** Include in next routine maintenance
            """)
    
    # Maintenance Calendar Visualization
    st.markdown("---")
    st.subheader("📅 Maintenance Calendar View")
    
    # Create calendar view
    calendar_data = pd.DataFrame({
        'Date': pd.date_range(start=today, periods=90, freq='D'),
        'Maintenance_Count': np.random.poisson(0.5, 90)  # Random maintenance events
    })
    
    # Add scheduled maintenance
    for _, row in maintenance_schedule.iterrows():
        if row['Predicted_Failure'] in calendar_data['Date'].values:
            idx = calendar_data[calendar_data['Date'] == row['Predicted_Failure']].index[0]
            calendar_data.loc[idx, 'Maintenance_Count'] += 1
    
    st.line_chart(calendar_data.set_index('Date')['Maintenance_Count'])

# --- 11. ENERGY & ESG SUSTAINABILITY (Enhanced with Formulas) ---
elif menu == "⚡ Energy & ESG Sustainability":
    st.markdown('<div class="main-header"><h1>⚡ Energy Optimization & ESG Sustainability Dashboard</h1><p>Monitor energy consumption and environmental impact</p></div>', unsafe_allow_html=True)
    
    # Energy Metrics
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    
    with col_e1:
        total_daily_energy = fleet['Daily_Energy_Cost_Rp'].sum()
        st.markdown(f'''
        <div class="metric-card">
            <h4>💰 Daily Energy Cost</h4>
            <h2>Rp {total_daily_energy:,.0f}</h2>
            <p>Monthly: Rp {total_daily_energy * 30:,.0f}</p>
            <small>Formula: Σ(Operational Hours × Power Consumption × 1500)</small>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_e2:
        avg_efficiency = 82.4  # Simulated
        st.markdown(f'''
        <div class="metric-card">
            <h4>⚡ Energy Efficiency</h4>
            <h2>{avg_efficiency}%</h2>
            <p>vs Standard: 68% | +14.4%</p>
            <small>Formula: (Actual Output / Maximum Possible Output) × 100%</small>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_e3:
        co2_reduction = 152.8
        st.markdown(f'''
        <div class="metric-card">
            <h4>🌿 CO2 Reduction</h4>
            <h2>{co2_reduction} Ton</h4>
            <p>Year 2024 | Target: 120 Ton</p>
            <small>Formula: (Energy Saved × 0.85 kg CO2/kWh) / 1000</small>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_e4:
        water_saved = 12500  # Liters
        st.markdown(f'''
        <div class="metric-card">
            <h4>💧 Water Saved</h4>
            <h2>{water_saved:,.0f} L</h2>
            <p>Monthly Average</p>
            <small>Dry compression technology saves 100% water vs water-cooled</small>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        st.subheader("Real-time Power Load Distribution (kWh)")
        
        # Simulate 24-hour power load with formula
        hours = list(range(24))
        base_load = 400
        peak_multiplier = [1.0, 0.8, 0.7, 0.6, 0.6, 0.7, 0.9, 1.2, 1.5, 
                          1.8, 2.0, 2.2, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 
                          1.6, 1.4, 1.2, 1.1, 1.0, 0.9]
        
        power_load = [base_load * mult for mult in peak_multiplier]
        
        power_df = pd.DataFrame({
            'Hour': hours,
            'Power_kW': power_load,
            'Cost_Rp': [p * 1500 for p in power_load]
        })
        
        # Create matplotlib chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hours, power_load, marker='o', color='#005293', linewidth=3)
        ax.fill_between(hours, power_load, alpha=0.3, color='#005293')
        
        # Find and mark peak hour
        peak_hour = power_load.index(max(power_load))
        ax.annotate(f'Peak: {max(power_load):.0f} kW',
                   xy=(peak_hour, max(power_load)),
                   xytext=(peak_hour, max(power_load) + 20),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, color='red')
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Power Consumption (kW)')
        ax.set_title('24-Hour Power Load Profile')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
        
        st.pyplot(fig)
        
        st.info(f"""
        **AI Energy Optimization Insight:**
        
        Peak load occurs at **{peak_hour}:00** with **{max(power_load):.0f} kW** consumption.
        
        **Recommendation:**
        - Load shifting: Move non-critical operations to off-peak hours (22:00-06:00)
        - Potential savings: **15%** (Rp {max(power_load) * 1500 * 0.15:,.0f}/day)
        - Implement smart scheduling for compressor units
        """)
    
    with col_ch2:
        st.subheader("ESG: Carbon Footprint Analysis")
        
        # CO2 Calculation Breakdown
        monthly_energy_kwh = total_daily_energy * 30 / 1500  # Convert Rp to kWh
        
        co2_data = pd.DataFrame({
            'Category': ['Direct Emissions', 'Indirect Emissions', 'Avoided Emissions', 'Net Footprint'],
            'CO2_Tons': [25.3, 18.7, -152.8, -108.8],
            'Color': ['#ef4444', '#f59e0b', '#10b981', '#005293']
        })
        
        # Create matplotlib bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(co2_data['Category'], co2_data['CO2_Tons'], 
                     color=co2_data['Color'])
        
        # Add value labels on bars
        for bar, value in zip(bars, co2_data['CO2_Tons']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f} Ton', ha='center', va='bottom' if height > 0 else 'top')
        
        ax.set_ylabel('CO2 (Tons)')
        ax.set_title('Carbon Footprint Breakdown (Monthly)')
        ax.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
        
        # ESG Score Calculation
        st.markdown("""
        **ESG Score Calculation:**
        
        | Metric | Weight | Score | Contribution |
        |--------|--------|-------|--------------|
        | Energy Efficiency | 30% | 92/100 | 27.6 |
        | Carbon Reduction | 40% | 88/100 | 35.2 |
        | Water Conservation | 20% | 95/100 | 19.0 |
        | Circular Economy | 10% | 85/100 | 8.5 |
        | **Total ESG Score** | **100%** | **90.3/100** | **Excellent** |
        """)
    
    # Energy Savings Calculation
    st.markdown("---")
    st.subheader("💡 Energy Savings Calculator")
    
    col_calc1, col_calc2, col_calc3 = st.columns(3)
    
    with col_calc1:
        current_hours = st.number_input("Current Daily Operating Hours", 
                                       min_value=8.0, max_value=24.0, value=16.0, step=0.5)
    
    with col_calc2:
        current_power = st.number_input("Current Power Consumption (kW)", 
                                       min_value=50.0, max_value=200.0, value=100.0, step=5.0)
    
    with col_calc3:
        optimization_rate = st.slider("Optimization Potential (%)", 0, 30, 15)
    
    # Calculate savings
    current_daily = current_hours * current_power * 1500
    optimized_hours = current_hours * (1 - optimization_rate/100)
    optimized_daily = optimized_hours * current_power * 1500 * 0.9  # Assume 10% efficiency gain
    
    daily_savings = current_daily - optimized_daily
    monthly_savings = daily_savings * 30
    annual_savings = monthly_savings * 12
    
    st.success(f"""
    **Potential Savings Calculation:**
    
    - Current Daily Cost: **Rp {current_daily:,.0f}**
    - Optimized Daily Cost: **Rp {optimized_daily:,.0f}**
    - Daily Savings: **Rp {daily_savings:,.0f}**
    - Monthly Savings: **Rp {monthly_savings:,.0f}**
    - Annual Savings: **Rp {annual_savings:,.0f}**
    
    *Based on {optimization_rate}% optimization through load shifting and efficiency improvements*
    """)

# --- 12. FINANCIAL LOSS & ROI ANALYSIS (Enhanced with Detailed Formulas) ---
elif menu == "💰 Financial Loss & ROI Analysis":
    st.markdown('<div class="main-header"><h1>💰 Financial Exposure & ROI Control Center</h1><p>Comprehensive financial analysis and risk quantification</p></div>', unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.subheader("📉 Loss Exposure Calculation")
        
        # Input Parameters with explanations
        st.markdown("#### 📊 Input Parameters")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            downtime_cost = st.number_input("Cost of Downtime (Rp/Hour)", 
                                          value=25000000,
                                          help="Includes lost production, labor costs, and penalty fees")
            
            repair_cost = st.number_input("Average Repair Cost (Rp)", 
                                        value=15000000,
                                        help="Average cost of emergency repairs including parts and labor")
        
        with col_p2:
            risk_probability = st.slider("Risk Probability (%)", 
                                       0, 100, 30,
                                       help="Probability of failure occurring based on health score and usage")
            
            affected_hours = st.number_input("Expected Downtime Hours", 
                                           min_value=1, max_value=168, value=48,
                                           help="Estimated downtime duration in case of failure")
        
        # CALCULATION FORMULAS (EXPLICIT)
        st.markdown("---")
        st.markdown("#### 🧮 Calculation Breakdown")
        
        # Formula 1: Direct Loss
        direct_loss = downtime_cost * affected_hours + repair_cost
        
        # Formula 2: Indirect Loss (20% of direct loss)
        indirect_loss = direct_loss * 0.20
        
        # Formula 3: Total Potential Loss
        total_potential_loss = direct_loss + indirect_loss
        
        # Formula 4: Risk-adjusted Loss
        risk_adjusted_loss = total_potential_loss * (risk_probability / 100)
        
        st.markdown(f"""
        **1. Direct Loss Calculation:**
        ```
        Direct Loss = (Downtime Cost × Hours) + Repair Cost
                    = (Rp {downtime_cost:,.0f} × {affected_hours}) + Rp {repair_cost:,.0f}
                    = Rp {direct_loss:,.0f}
        ```
        
        **2. Indirect Loss Calculation:**
        ```
        Indirect Loss = Direct Loss × 20%
                      = Rp {direct_loss:,.0f} × 0.20
                      = Rp {indirect_loss:,.0f}
        ```
        
        **3. Total Potential Loss:**
        ```
        Total Loss = Direct Loss + Indirect Loss
                   = Rp {direct_loss:,.0f} + Rp {indirect_loss:,.0f}
                   = Rp {total_potential_loss:,.0f}
        ```
        
        **4. Risk-adjusted Loss (Probability {risk_probability}%):**
        ```
        Risk-adjusted Loss = Total Loss × (Risk Probability / 100)
                          = Rp {total_potential_loss:,.0f} × ({risk_probability}/100)
                          = Rp {risk_adjusted_loss:,.0f}
        ```
        """)
        
        # Display Results
        st.markdown(f"""
        <div style="background-color: #fef2f2; padding: 20px; border-radius: 10px; border-left: 5px solid #ef4444;">
            <h3 style="color: #dc2626; margin-top: 0;">⚠️ FINANCIAL EXPOSURE</h3>
            <h2 style="color: #dc2626;">Rp {risk_adjusted_loss:,.0f}</h2>
            <p>Potential loss that could be avoided with preventive maintenance</p>
            <small>Based on {risk_probability}% failure probability</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("💡 **Admin Insight:** This amount represents the financial risk exposure that can be mitigated through predictive maintenance.")
    
    with col_f2:
        st.subheader("📈 ROI Analysis")
        
        st.markdown("#### 💰 Service ROI Model: Sigma Air Utility")
        
        # ROI Calculation Inputs
        st.markdown("**Investment Parameters:**")
        
        col_roi1, col_roi2 = st.columns(2)
        with col_roi1:
            capex_savings = st.number_input("CAPEX Savings (Rp)", 
                                          value=1500000000,
                                          help="Capital expenditure avoided through pay-per-use model")
            
            energy_savings = st.number_input("Monthly Energy Savings (Rp)", 
                                           value=12000000,
                                           help="Reduced energy consumption through efficiency")
        
        with col_roi2:
            maintenance_savings = st.number_input("Monthly Maintenance Savings (Rp)", 
                                                value=8000000,
                                                help="Reduced maintenance costs")
            
            admin_fee = st.slider("Admin Fee Optimization (%)", 0, 30, 15,
                                help="Reduction in administrative costs")
        
        # ROI Calculation Formulas
        st.markdown("---")
        st.markdown("#### 🧮 ROI Calculation")
        
        # Annual Savings Calculation
        annual_energy_savings = energy_savings * 12
        annual_maintenance_savings = maintenance_savings * 12
        admin_savings = (capex_savings * 0.15) * (admin_fee / 100)  # Simplified
        
        total_annual_savings = annual_energy_savings + annual_maintenance_savings + admin_savings
        
        # ROI Formula
        roi_percentage = (total_annual_savings / capex_savings) * 100
        
        # Payback Period
        payback_months = (capex_savings / total_annual_savings) * 12
        
        st.markdown(f"""
        **1. Annual Savings Breakdown:**
        ```
        Energy Savings = Monthly Savings × 12
                       = Rp {energy_savings:,.0f} × 12
                       = Rp {annual_energy_savings:,.0f}/year
        
        Maintenance Savings = Monthly Savings × 12
                            = Rp {maintenance_savings:,.0f} × 12
                            = Rp {annual_maintenance_savings:,.0f}/year
        
        Admin Fee Savings = CAPEX × 15% × {admin_fee}%
                          = Rp {capex_savings:,.0f} × 0.15 × {admin_fee/100}
                          = Rp {admin_savings:,.0f}/year
        
        Total Annual Savings = Rp {total_annual_savings:,.0f}/year
        ```
        
        **2. ROI Calculation:**
        ```
        ROI = (Total Annual Savings / CAPEX Savings) × 100%
            = (Rp {total_annual_savings:,.0f} / Rp {capex_savings:,.0f}) × 100%
            = {roi_percentage:.1f}%
        ```
        
        **3. Payback Period:**
        ```
        Payback Period = (CAPEX Savings / Total Annual Savings) × 12 months
                       = (Rp {capex_savings:,.0f} / Rp {total_annual_savings:,.0f}) × 12
                       = {payback_months:.1f} months
        ```
        """)
        
        # Display ROI Results
        st.markdown(f"""
        <div style="background-color: #f0fdf4; padding: 20px; border-radius: 10px; border-left: 5px solid #10b981;">
            <h3 style="color: #059669; margin-top: 0;">📊 ROI SUMMARY</h3>
            <h2 style="color: #059669;">{roi_percentage:.1f}% ROI</h2>
            <p>Payback Period: {payback_months:.1f} months</p>
            <h4 style="color: #059669;">Rp {total_annual_savings:,.0f}</h4>
            <p>Total Value Gained (First Year)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ROI Visualization with matplotlib
        years = [1, 2, 3, 4, 5]
        cumulative_savings = [total_annual_savings * y for y in years]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(years, cumulative_savings, marker='o', color='#10b981', 
                linewidth=3, label='Cumulative Savings')
        ax.axhline(y=capex_savings, color='#ef4444', linestyle='--', 
                  linewidth=2, label='Initial Investment')
        
        ax.set_xlabel('Years')
        ax.set_ylabel('Amount (Rp)')
        ax.set_title('5-Year ROI Projection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show in millions
        def millions(x, pos):
            return f'Rp {x/1e6:.0f}M'
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(millions))
        
        st.pyplot(fig)
    
    # Additional Financial Analysis
    st.markdown("---")
    st.subheader("📊 Comparative Financial Analysis")
    
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    
    with col_comp1:
        st.markdown("""
        **Traditional Model:**
        - CAPEX: Rp 2.5M
        - Monthly Opex: Rp 45jt
        - Downtime: 7%
        - Total 5-Year Cost: **Rp 3.1M**
        """)
    
    with col_comp2:
        st.markdown("""
        **Sigma Air Utility (Current):**
        - CAPEX: Rp 1.0M
        - Monthly Opex: Rp 32jt
        - Downtime: 2%
        - Total 5-Year Cost: **Rp 2.0M**
        """)
    
    with col_comp3:
        savings = 3100000000 - 2000000000
        st.markdown(f"""
        **Savings with Kaeser:**
        - CAPEX Savings: **35%**
        - OPEX Savings: **29%**
        - Downtime Reduction: **71%**
        - **Total 5-Year Savings: Rp {savings:,.0f}**
        """)
    
    # Export Financial Report
    st.markdown("---")
    st.subheader("📥 Export Financial Report")
    
    # Create comprehensive report
    report_data = {
        'Parameter': [
            'Cost of Downtime per Hour',
            'Average Repair Cost',
            'Risk Probability',
            'Expected Downtime Hours',
            'Direct Loss',
            'Indirect Loss',
            'Total Potential Loss',
            'Risk-adjusted Loss',
            'CAPEX Savings',
            'Monthly Energy Savings',
            'Monthly Maintenance Savings',
            'Total Annual Savings',
            'ROI Percentage',
            'Payback Period (months)'
        ],
        'Value': [
            f'Rp {downtime_cost:,.0f}',
            f'Rp {repair_cost:,.0f}',
            f'{risk_probability}%',
            f'{affected_hours} hours',
            f'Rp {direct_loss:,.0f}',
            f'Rp {indirect_loss:,.0f}',
            f'Rp {total_potential_loss:,.0f}',
            f'Rp {risk_adjusted_loss:,.0f}',
            f'Rp {capex_savings:,.0f}',
            f'Rp {energy_savings:,.0f}',
            f'Rp {maintenance_savings:,.0f}',
            f'Rp {total_annual_savings:,.0f}',
            f'{roi_percentage:.1f}%',
            f'{payback_months:.1f}'
        ]
    }
    
    report_df = pd.DataFrame(report_data)
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # CSV Export
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Financial Report (CSV)",
            data=csv,
            file_name=f"Kaeser_Financial_Analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        # Summary PDF (simulated)
        st.download_button(
            label="📑 Download Executive Summary (PDF)",
            data=csv,  # In real implementation, generate actual PDF
            file_name=f"Kaeser_Executive_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# --- 13. FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 20px;">
    <p>© 2024 Kaeser Kompressoren SE | Smart Enterprise AI Dashboard v2.1</p>
    <p>This system uses machine learning for predictive maintenance and energy optimization</p>
    <p>Last Updated: """ + datetime.now().strftime("%d %B %Y %H:%M:%S") + """</p>
</div>
""", unsafe_allow_html=True)