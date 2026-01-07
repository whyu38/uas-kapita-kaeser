from tkinter import N
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io

# --- 1. CONFIG ---
st.set_page_config(page_title="Kaeser Smart-Enterprise AI", page_icon="🏢", layout="wide")

# --- 2. CSS CUSTOM (Warna Teks Fix & Card Hover) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #F4F7F9; }
    
    [data-testid="stSidebar"] { background-color: #002B49; border-right: 1px solid #e0e0e0; }
    [data-testid="stSidebar"] * { color: white !important; }

    .ux-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-top: 5px solid #005293;
        margin-bottom: 20px;
    }
    .ux-card h4 { color: #64748b !important; font-size: 0.85rem; margin-bottom: 8px; font-weight: 600; text-transform: uppercase; }
    .ux-card h2 { color: #1e293b !important; font-size: 1.8rem; font-weight: 800; margin: 0; }
    .ux-card p { color: #94a3b8 !important; font-size: 0.8rem; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DUMMY DATA GENERATOR ---
@st.cache_data
def load_enterprise_data():
    n_units = 20
    # 1. Bangkitkan Health Score terlebih dahulu
    health_scores = np.random.randint(40, 100, n_units).tolist()
    
    # 2. Tentukan Status berdasarkan Health Score (Logika Enterprise)
    statuses = []
    for score in health_scores:
        if score < 60:
            statuses.append('Critical')
        elif score < 85:
            statuses.append('Warning')
        else:
            statuses.append('Healthy')
            
    random_days = np.random.randint(10, 200, n_units).tolist()
    
    fleet_data = pd.DataFrame({
        'Unit_ID': [f'K-DX-{i:03}' for i in range(1, n_units + 1)],
        'Lokasi': np.random.choice(['Jakarta', 'Surabaya', 'Bandung', 'Semarang'], n_units),
        'Status': statuses,
        'Health_Score': health_scores,
        'Last_Service': [datetime.now() - timedelta(days=int(x)) for x in random_days],
        'lat': np.random.uniform(-7.5, -6.0, n_units),
        'lon': np.random.uniform(106.5, 113.0, n_units)
    })
    return fleet_data

fleet = load_enterprise_data()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("C:\Web Programming\Kuliah\kaeser-project\kaeser-logo.png", width=200)
    st.markdown("---")
    menu = st.radio("Enterprise Navigation", [
        "🌐 Fleet Management Control",
        "📊 Executive Performance & Export",
        "🧠 AI Diagnostic Laboratory",
        "📅 Smart Maintenance Calendar",
        "⚡ Energy & ESG Sustainability",
        "💎 Financial Loss & ROI Analysis"
    ])
    st.markdown("---")
    st.info("**Admin Node:** Jakarta-Central\n\n**System Time:** " + datetime.now().strftime("%H:%M:%S"))

# --- 5. MODULES ---

# SCREEN 1: FLEET CONTROL
if menu == "🌐 Fleet Management Control":
    st.title("🌐 Global Fleet Control Center")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Real-time Unit Distribution")
        st.map(fleet)
    
    with col2:
        st.subheader("Regional Quick Stats")
        selected_loc = st.selectbox("Filter Wilayah", ["All Regions"] + list(fleet['Lokasi'].unique()))
        df_filtered = fleet if selected_loc == "All Regions" else fleet[fleet['Lokasi'] == selected_loc]
        
        st.write(f"Menampilkan {len(df_filtered)} unit di {selected_loc}")
        c1, c2 = st.columns(2)
        c1.metric("Warning", len(df_filtered[df_filtered['Status'] == 'Warning']))
        c2.metric("Critical", len(df_filtered[df_filtered['Status'] == 'Critical']), delta_color="inverse")
        
        st.write("---")
        for i, row in df_filtered.iterrows():
            with st.expander(f"🆔 {row['Unit_ID']} - {row['Status']}"):
                st.write(f"**Health:** {row['Health_Score']}%")
                st.write(f"**Lokasi:** {row['Lokasi']}")
                st.progress(row['Health_Score'] / 100)

# SCREEN 2: EXECUTIVE PERFORMANCE
elif menu == "📊 Executive Performance & Export":
    st.title("📊 Enterprise Performance Analysis")
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown('<div class="ux-card"><h4>System Reliability</h4><h2>99.92%</h2><p>vs 98.5% Target</p></div>', unsafe_allow_html=True)
    with k2: st.markdown('<div class="ux-card"><h4>Avg Health Score</h4><h2>84/100</h2><p>Stable Trend</p></div>', unsafe_allow_html=True)
    with k3: st.markdown('<div class="ux-card"><h4>Cost Efficiency</h4><h2>+12.4%</h2><p>Reduction in Waste</p></div>', unsafe_allow_html=True)
    with k4: st.markdown('<div class="ux-card"><h4>Units Online</h4><h2>19/20</h2><p>1 Unit Maintenance</p></div>', unsafe_allow_html=True)

    st.subheader("Monthly Efficiency Comparison (Current Year)")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data_y = np.random.randint(75, 95, 12)
    df_perf = pd.DataFrame({'Bulan': months, 'Efficiency_Score': data_y})
    
    st.line_chart(df_perf.set_index('Bulan'))
    
    st.write("---")
    st.subheader("📥 Export Center")
    csv = df_perf.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Performance Report (.CSV)", data=csv, file_name="Kaeser_Performance_2024.csv", mime='text/csv')

# SCREEN 3: AI DIAGNOSTIC
elif menu == "🧠 AI Diagnostic Laboratory":
    st.title("🧠 AI Diagnostic: Deep Anomaly Analysis")
    
    f1, f2, f3 = st.columns(3)
    loc_f = f1.selectbox("Filter Wilayah", fleet['Lokasi'].unique())
    unit_f = f2.selectbox("Select Unit ID", fleet[fleet['Lokasi'] == loc_f]['Unit_ID'])
    sens = f3.slider("AI Sensitivity Threshold", 0.01, 0.20, 0.05)

    # ML Logic
    n = 300
    df_diag = pd.DataFrame({
        'Suhu': np.r_[np.random.normal(70, 2, n), np.random.normal(98, 4, 15)],
        'Tekanan': np.r_[np.random.normal(7, 0.3, n), np.random.normal(4.5, 1.2, 15)]
    })
    model = IsolationForest(contamination=sens, random_state=42)
    df_diag['Prediksi'] = model.fit_predict(df_diag[['Suhu', 'Tekanan']])
    df_diag['Label'] = np.where(df_diag['Prediksi'] == -1, 'Anomali', 'Normal')

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(f"Sensor Mapping: {unit_f}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_diag, x='Suhu', y='Tekanan', hue='Label', palette={'Normal': '#005293', 'Anomali': '#ef4444'}, ax=ax)
        st.pyplot(fig)
    
    with c2:
        st.subheader("AI Insight")
        anoms = len(df_diag[df_diag['Prediksi'] == -1])
        if anoms > 5:
            st.error(f"🔴 KRITIS: Terdeteksi Gejala Kebocoran (Leakage) pada {unit_f}")
            st.write("**Indikasi:** Tekanan drop drastis namun suhu motor meningkat. Ini tanda kebocoran udara pada pipa distribusi utama.")
        else:
            st.success("🟢 NORMAL: Komponen katup dan motor berjalan sinkron.")

# SCREEN 4: PREDICTIVE MAINTENANCE
elif menu == "📅 Smart Maintenance Calendar":
    st.title("📅 Predictive Maintenance Planner")
    st.info("AI memprediksi kegagalan komponen berdasarkan 'Wear-Tear Pattern' sensor.")
    
    schedule = pd.DataFrame({
        'Unit': ['K-DX-004', 'K-DX-012', 'K-DX-007', 'K-DX-019'],
        'Wilayah': ['Jakarta', 'Surabaya', 'Semarang', 'Jakarta'],
        'Komponen': ['Air Filter', 'Oil Separator', 'Motor Bearing', 'Coupling'],
        'Prediksi_Gagal': ['2024-06-20', '2024-07-05', '2024-12-15', '2025-01-05'],
        'Risk_Impact': ['High', 'Medium', 'Critical', 'Low']
    })
    
    st.table(schedule)
    
    st.subheader("Risk Analysis")
    selected_unit = st.selectbox("Detail Unit", schedule['Unit'])
    risk = schedule[schedule['Unit'] == selected_unit].iloc[0]
    
    if risk['Risk_Impact'] == 'Critical':
        st.error(f"⚠️ **URGENT:** Jika {risk['Komponen']} pada {risk['Unit']} tidak diganti sebelum {risk['Prediksi_Gagal']}, potensi kerusakan total motor induksi dapat menyebabkan downtime 48 jam senilai Rp 120jt.")
    else:
        st.warning(f"💡 **Saran:** Lakukan penggantian {risk['Komponen']} pada kunjungan rutin berikutnya.")

# SCREEN 5: ENERGY & ESG
elif menu == "⚡ Energy & ESG Sustainability":
    st.title("⚡ Energy Optimization & ESG Report")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Real-time Power Load (kWh)")
        st.bar_chart(np.random.normal(450, 40, 24))
        st.write("**Analisis:** Penggunaan listrik memuncak pada pukul 14:00. AI merekomendasikan penyesuaian beban kerja motor (Load-shifting) untuk hemat 15%.")
    
    with col2:
        st.subheader("ESG: Carbon Footprint Tracker")
        st.markdown('<div class="ux-card"><h4>Total CO2 Reduction</h4><h2>152.8 Ton</h2><p>Tahun 2024</p></div>', unsafe_allow_html=True)
        st.write("---")
        st.success("Unit ini berkontribusi pada target Net-Zero pabrik dengan efisiensi Sigma Air Utility sebesar 22% lebih baik dari kompresor standar.")

# SCREEN 6: ROI & LOSS
elif menu == "💎 Financial Loss & ROI Analysis":
    st.title("💎 Financial Exposure & ROI Control")
    st.write("Modul ini menghitung potensi kerugian finansial akibat inefisiensi atau kerusakan.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Loss Exposure Calculation")
        downtime_rate = st.number_input("Cost of Downtime (Rp/Jam)", value=25000000)
        risk_probability = st.slider("Risk Probability (%)", 0, 100, 30)
        potential_loss = downtime_rate * (risk_probability/100) * 24
        
        st.metric("Potential Loss/Day", f"Rp {potential_loss:,.0f}", delta="Risk Exposure")
        st.caption("Admin Insight: Angka ini adalah biaya yang bisa dihindari jika Predictive Maintenance dilakukan hari ini.")

    with col2:
        st.subheader("Service ROI Estimate")
        st.write("**Model: Sigma Air Utility (Pay-per-use)**")
        st.write("- Penghematan CAPEX: Rp 1.5M")
        st.write("- Penghematan Energi: Rp 12jt/Bulan")
        st.write("- Admin Fee Optimization: 15%")
        st.markdown('<div class="ux-card"><h4>Total Value Gained</h4><h2>Rp 450jt</h2><p>Tahun Pertama</p></div>', unsafe_allow_html=True)