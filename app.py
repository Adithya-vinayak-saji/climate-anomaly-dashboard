import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import pymannkendall as mk

# --- 0. INITIALIZATION ---
st.set_page_config(layout="wide", page_title="Climate Intelligence Suite", page_icon="🌍")

# --- PROFESSIONAL UI STYLING (The "Smoothness") ---
st.markdown("""
    <style>
    /* Global Background */
    .stApp { background-color: #f1f5f9; }
    
    /* Smooth Fade-In for all components */
    .stCard, .stTab, .main-container {
        animation: fadeIn 0.8s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Professional Landing Page Hero */
    .hero-section {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 60px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        border-bottom: 4px solid #3b82f6;
        height: 100%;
        transition: transform 0.3s ease;
    }
    .feature-card:hover { transform: translateY(-5px); }

    /* Summary Stats Box */
    .summary-box { 
        background: white; padding: 20px; border-radius: 15px; 
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-left: 5px solid #3b82f6;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CORE LOGIC (UNTOUCHED) ---
def calculate_es(temp_celsius):
    return 6.112 * np.exp((17.67 * temp_celsius) / (temp_celsius + 243.5))

def find_best_column(columns, keywords):
    for col in columns:
        for kw in keywords:
            if kw.lower() in str(col).lower(): return col
    return columns[0]

def get_ai_insight(api_key, prompt_context):
    if not api_key: return "🤖 AI: Please enter API Key in the sidebar."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f"Explain climate trends simply: {prompt_context}")
        return response.text
    except Exception as e: return f"🤖 AI Error: {str(e)}"

# --- 1. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Control Center")
    st.markdown("---")
    user_api_key = st.text_input("Gemini API Key", type="password", help="For AI-driven trend interpretation.")
    
    st.header("📂 Data Ingestion")
    file1 = st.file_uploader("Primary Dataset (Excel)", type=["xlsx"])
    file2 = st.file_uploader("Comparison Dataset (Optional)", type=["xlsx"])
    
    if file1:
        xl1 = pd.ExcelFile(file1)
        base_sheet = st.selectbox("Baseline Sheet", xl1.sheet_names)
        anom_sheet = st.selectbox("Anomaly Sheet", xl1.sheet_names, index=1 if len(xl1.sheet_names) > 1 else 0)

# --- 2. MAIN INTERFACE ---
if not file1:
    # --- PROFESSIONAL LANDING PAGE (Empty State) ---
    st.markdown("""
        <div class="hero-section">
            <h1>🌍 Climate Intelligence Suite</h1>
            <p style="font-size: 1.2rem; opacity: 0.8;">High-Precision Anomaly Detection & Atmospheric Physics Analysis</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="feature-card"><h3>📉 Trend Analysis</h3>
        Statistical detection of climate patterns using <b>Mann-Kendall</b> tests and <b>Sen's Slope</b> for robust trend quantification.</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="feature-card"><h3>🌡️ Physics Engine</h3>
        Interactive <b>Clausius-Clapeyron</b> relationship modeling to calculate atmospheric moisture capacity gains.</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="feature-card"><h3>🤖 AI Synthesis</h3>
        Generative AI integration to translate complex statistical markers into actionable regional insights.</div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("💡 **Getting Started:** Please upload your Excel data in the sidebar to begin the analysis.")
    
    # Static Image for Professional Look
    

else:
    # --- DATA-DRIVEN DASHBOARD ---
    df_base = pd.read_excel(file1, sheet_name=base_sheet)
    df_anom = pd.read_excel(file1, sheet_name=anom_sheet)
    
    numeric_cols = df_anom.select_dtypes(include=[np.number]).columns.tolist()
    primary_var = st.selectbox("Select Target Metric", numeric_cols)
    
    time_col = find_best_column(df_anom.columns, ["date", "year", "time"])
    if time_col not in df_anom.columns:
        df_anom = df_anom.reset_index().rename(columns={'index': 'Index'})
        time_col = 'Index'

    # Statistics Calculations
    mk_res = mk.original_test(df_anom[primary_var])
    slope, tau = getattr(mk_res, 'slope', 0), getattr(mk_res, 'tau', 0)

    # Summary Header
    st.markdown(f"""
        <div class="summary-box">
            <h2 style="margin:0; color:#1e293b;">📊 Analysis Overview: {primary_var}</h2>
            <p style="margin:5px 0 0 0; color:#64748b;">
                <b>Trend:</b> {mk_res.trend.upper()} &nbsp; | &nbsp; 
                <b>Sen's Slope:</b> {slope:.4f} &nbsp; | &nbsp; 
                <b>Kendall Tau:</b> {tau:.4f}
            </p>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📉 Timeline", "🌡️ C-C Relation", "📈 Statistics", "🏠 AI Impact", "🌐 Regional Comp"])

    with tab1:
        st.subheader("Temporal Distribution")
        fig_ts = px.line(df_anom, x=time_col, y=primary_var, markers=True, template="plotly_white", color_discrete_sequence=['#3b82f6'])
        fig_ts.update_layout(hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)

    with tab2:
        st.subheader("Atmospheric Moisture Capacity Gain")
        c1, c2 = st.columns(2)
        with c1:
            base_temp_col = st.selectbox("Select Baseline Temp", df_base.select_dtypes(include=[np.number]).columns)
            static_base = df_base[base_temp_col].mean()
        with c2:
            cc_anom_col = st.selectbox("Select Anomaly Temp", numeric_cols)
        
        df_anom['m_gain'] = df_anom[cc_anom_col].apply(
            lambda x: ((calculate_es(static_base + x) - calculate_es(static_base)) / calculate_es(static_base)) * 100
        )

        fig_cc = px.bar(df_anom, x=time_col, y='m_gain', color='m_gain', 
                        color_continuous_scale='RdBu_r', template="plotly_white",
                        labels={'m_gain': 'Moisture Gain (%)'})
        st.plotly_chart(fig_cc, use_container_width=True)
        
        st.markdown("> **Note:** This calculation uses the Clausius-Clapeyron equation, showing approximately a 7% increase in moisture capacity per 1°C of warming.")
        

    with tab3:
        st.subheader("Statistical Precision Models")
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.dataframe(pd.DataFrame({
                "Metric": ["Trend Result", "P-Value", "Sen's Slope", "Tau"],
                "Value": [mk_res.trend, f"{mk_res.p:.4f}", f"{slope:.4f}", f"{tau:.4f}"]
            }))
        with col_s2:
            fig_stat = px.scatter(df_anom, x=time_col, y=primary_var, trendline="ols", 
                                  title="OLS Regression vs. Mann-Kendall Trend", template="plotly_white")
            st.plotly_chart(fig_stat, use_container_width=True)

    with tab4:
        st.subheader("AI-Driven Interpretation")
        if st.button("✨ Synthesize Dataset Insights"):
            with st.spinner("AI analyzing trend vectors..."):
                insight = get_ai_insight(user_api_key, f"Trend: {mk_res.trend}, Slope: {slope}, Var: {primary_var}")
                st.info(insight)

    with tab5:
        if file2:
            df_comp = pd.read_excel(file2)
            comp_var = st.selectbox("Comparison Metric", df_comp.select_dtypes(include=[np.number]).columns)
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=df_anom[time_col], y=df_anom[primary_var], name="Region A", line=dict(color='#3b82f6')))
            fig_comp.add_trace(go.Scatter(x=df_anom[time_col], y=df_comp[comp_var], name="Region B", line=dict(color='#ef4444', dash='dash')))
            fig_comp.update_layout(template="plotly_white", title="Cross-Regional Comparison")
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Please upload a second file to enable regional comparison.")

    st.markdown("---")
    st.download_button("📥 Export Analysis Report", df_anom.to_csv(), "climate_report.csv", "text/csv")