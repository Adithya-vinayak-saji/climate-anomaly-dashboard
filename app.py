import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import google.generativeai as genai
import pymannkendall as mk

# --- INITIALIZATION ---
st.set_page_config(layout="wide", page_title="Drought Intelligence Suite", page_icon="🌵")

if 'saved_key' not in st.session_state:
    st.session_state.saved_key = ""

# --- CORE MATH ENGINE (Logic Intact) ---
def calculate_simplified_pet(temp, wind, humidity):
    es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    ea = es * (humidity / 100)
    vpd = es - ea 
    pet = (0.0023 * (temp + 17.8) * (vpd**0.5) * (1 + 0.05 * wind)) * 10 
    return pet

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌵 Drought Control")
    input_key = st.text_input("Gemini API Key", value=st.session_state.saved_key, type="password")
    if st.checkbox("Remember API Key", value=bool(st.session_state.saved_key)):
        st.session_state.saved_key = input_key
    
    uploaded_file = st.file_uploader("Upload Climate Data (Excel)", type=["xlsx"])
    selected_sheet = None
    if uploaded_file:
        xl = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("Select the Sheet", xl.sheet_names)

# --- MAIN LOGIC ---
if uploaded_file and selected_sheet:
    df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    
    st.sidebar.subheader("Variable Mapping")
    all_cols = df_raw.columns.tolist()
    
    # Mapping for your 2-column format
    year_col = st.sidebar.selectbox("Year Column", all_cols, index=0)
    month_col = st.sidebar.selectbox("Month Column", all_cols, index=1)
    
    p_col = st.sidebar.selectbox("Precipitation (P)", all_cols, index=4)
    t_col = st.sidebar.selectbox("Temperature (T)", all_cols, index=2)
    w_col = st.sidebar.selectbox("Wind Speed (U)", all_cols, index=5)
    h_col = st.sidebar.selectbox("Humidity (RH)", all_cols, index=3)

    # --- DATA PROCESSING ---
    df = df_raw.copy()
    
    # 1. Date Stitching (Logic Intact)
    df['Combined_Date'] = pd.to_datetime(
        df[year_col].astype(str) + '-' + df[month_col].astype(str) + '-01', 
        errors='coerce'
    )
    
    # 2. Cleaning (Logic Intact)
    for col in [p_col, t_col, w_col, h_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Combined_Date', p_col, t_col]).sort_values('Combined_Date')

    # --- CALCULATIONS ---
    df['PET'] = df.apply(lambda r: calculate_simplified_pet(r[t_col], r[w_col], r[h_col]), axis=1)
    df['D'] = df[p_col] - df['PET']
    df['SPEI_Proxy'] = (df['D'] - df['D'].mean()) / df['D'].std()
    
    # Rolling Average Logic
    df['SPEI_Rolling'] = df['SPEI_Proxy'].rolling(window=12, center=True).mean()
    
    mk_res = mk.original_test(df['SPEI_Proxy'])

    st.title(f"🔍 SPEI Intelligence: {selected_sheet}")
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Trend", mk_res.trend.upper())
    with m2: st.metric("Sen's Slope", round(mk_res.slope, 5))
    with m3: st.metric("Confidence", f"{round((1-mk_res.p)*100, 2)}%")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Timeline", "⚖️ Water Balance", "🚨 Raw SPEI", "🌊 Smooth Trend", "🤖 AI Insight"
    ])

    # Reusable threshold lines for Tab 3 and Tab 4
    threshold_data = pd.DataFrame({'y': [-1.5, -2.0], 'label': ['Severe', 'Extreme']})
    danger_lines = alt.Chart(threshold_data).mark_rule(color='red', strokeDash=[5,5]).encode(
        y='y:Q',
        tooltip='label:N'
    )

    with tab1:
        base = alt.Chart(df).encode(x='Combined_Date:T')
        p_line = base.mark_line(color='#3b82f6').encode(y=alt.Y(f'{p_col}:Q', title='Precipitation'))
        pet_line = base.mark_line(color='#ef4444').encode(y='PET:Q')
        st.altair_chart((p_line + pet_line).properties(height=400).interactive(), use_container_width=True)

    with tab2:
        bars = alt.Chart(df).mark_bar().encode(
            x='Combined_Date:T',
            y='D:Q',
            color=alt.condition(alt.datum.D > 0, alt.value('#60a5fa'), alt.value('#f87171'))
        ).properties(height=400).interactive()
        st.altair_chart(bars, use_container_width=True)

    with tab3:
        # Restored Danger Lines for Raw Data
        spei_raw = alt.Chart(df).mark_line(point=True).encode(
            x='Combined_Date:T', y='SPEI_Proxy:Q', tooltip=['Combined_Date', 'SPEI_Proxy']
        ).properties(height=400)
        st.altair_chart((spei_raw + danger_lines).interactive(), use_container_width=True)

    with tab4:
        st.subheader("12-Month Rolling Average (Climate Signal)")
        smooth_chart = alt.Chart(df).mark_area(
            line={'color':'#1e40af'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#f87171', offset=0), 
                       alt.GradientStop(color='#60a5fa', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('Combined_Date:T', title='Timeline'),
            y=alt.Y('SPEI_Rolling:Q', title='Smoothed SPEI'),
            tooltip=['Combined_Date', 'SPEI_Rolling']
        ).properties(height=400)
        
        zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='black', strokeDash=[2,2]).encode(y='y')
        
        # Adding Danger Lines here too for context
        st.altair_chart((smooth_chart + zero_line + danger_lines).interactive(), use_container_width=True)

    with tab5:
        st.subheader("🤖 AI Strategic Briefing")
        if st.button("Analyze with Gemini"):
            if st.session_state.saved_key:
                try:
                    genai.configure(api_key=st.session_state.saved_key)
                    # Fixed ID to resolve 404
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = (f"Act as a climatologist. Analyzing trend: {mk_res.trend}, Slope: {mk_res.slope}. "
                              f"Briefly explain water security risks.")
                    with st.spinner("Analyzing..."):
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Connection Error: {e}")
            else:
                st.warning("Enter API Key in sidebar.")
else:
    st.info("👋 Upload data to generate 1970-2025 drought analysis.")