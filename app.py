import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import google.generativeai as genai
import pymannkendall as mk
from statsmodels.tsa.arima.model import ARIMA 

# --- INITIALIZATION ---
st.set_page_config(layout="wide", page_title="Drought Intelligence Suite", page_icon="🌵")

if 'saved_key' not in st.session_state:
    st.session_state.saved_key = ""

# --- CORE MATH ENGINE ---
def calculate_simplified_pet(temp, wind, humidity):
    es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    ea = es * (humidity / 100)
    vpd = es - ea 
    pet = (0.0023 * (temp + 17.8) * (vpd**0.5) * (1 + 0.05 * wind)) * 10 
    return pet

def get_longest_duration(series, threshold=-1.0):
    is_dry = series <= threshold
    if not is_dry.any(): return 0
    return (is_dry.groupby((~is_dry).cumsum()).cumcount()).max()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌵 Drought Control")
    input_key = st.text_input("Gemini API Key", value=st.session_state.saved_key, type="password")
    if st.checkbox("Remember API Key", value=bool(st.session_state.saved_key)):
        st.session_state.saved_key = input_key
    
    # --- NEW DIAGNOSTIC TOOL ---
    if st.button("🔍 Check API Connection"):
        if input_key:
            try:
                genai.configure(api_key=input_key)
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                st.success("API Key is working!")
                st.write("Your available models:")
                st.json(models)
            except Exception as e:
                st.error(f"Connection Failed: {e}")
        else:
            st.warning("Please enter a key first.")

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
    
    year_col = st.sidebar.selectbox("Year Column", all_cols, index=0)
    month_col = st.sidebar.selectbox("Month Column", all_cols, index=1)
    p_col = st.sidebar.selectbox("Precipitation (P)", all_cols, index=4)
    t_col = st.sidebar.selectbox("Temperature (T)", all_cols, index=2)
    w_col = st.sidebar.selectbox("Wind Speed (U)", all_cols, index=5)
    h_col = st.sidebar.selectbox("Humidity (RH)", all_cols, index=3)

    # --- DATA PROCESSING ---
    df = df_raw.copy()
    df['Combined_Date'] = pd.to_datetime(
        df[year_col].astype(str) + '-' + df[month_col].astype(str) + '-01', 
        errors='coerce'
    )
    for col in [p_col, t_col, w_col, h_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Combined_Date', p_col, t_col]).sort_values('Combined_Date')

    # --- CALCULATIONS ---
    df['PET'] = df.apply(lambda r: calculate_simplified_pet(r[t_col], r[w_col], r[h_col]), axis=1)
    df['D'] = df[p_col] - df['PET']
    df['SPEI_Proxy'] = (df['D'] - df['D'].mean()) / df['D'].std()
    df['SPEI_Rolling'] = df['SPEI_Proxy'].rolling(window=12, center=True).mean()
    
    extreme_count = len(df[df['SPEI_Proxy'] <= -2.0])
    max_duration = get_longest_duration(df['SPEI_Proxy'])
    mk_res = mk.original_test(df['SPEI_Proxy'])

    st.title(f"🔍 SPEI Intelligence: {selected_sheet}")
    
    # Metrics Header
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Trend", mk_res.trend.upper())
    with m2: st.metric("Sen's Slope", f"{mk_res.slope:.6f}") 
    with m3: st.metric("Extreme Events", extreme_count)
    with m4: st.metric("Max Duration", f"{max_duration} Months")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📉 Timeline", "⚖️ Water Balance", "🚨 Raw SPEI", "🌊 Smooth Trend", "🤖 AI Insight", "🔮 12-Month Forecast"
    ])

    threshold_data = pd.DataFrame({'y': [-1.5, -2.0], 'label': ['Severe', 'Extreme']})
    danger_lines = alt.Chart(threshold_data).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q', tooltip='label:N')

    # Chart sections for Tabs 1-4
    with tab1:
        base = alt.Chart(df).encode(x='Combined_Date:T')
        p_line = base.mark_line(color='#3b82f6').encode(y=alt.Y(f'{p_col}:Q', title='Precipitation'))
        pet_line = base.mark_line(color='#ef4444').encode(y='PET:Q')
        st.altair_chart((p_line + pet_line).properties(height=400).interactive(), use_container_width=True)

    with tab2:
        bars = alt.Chart(df).mark_bar().encode(
            x='Combined_Date:T', y='D:Q',
            color=alt.condition(alt.datum.D > 0, alt.value('#60a5fa'), alt.value('#f87171'))
        ).properties(height=400).interactive()
        st.altair_chart(bars, use_container_width=True)

    with tab3:
        spei_raw = alt.Chart(df).mark_line(point=True).encode(
            x='Combined_Date:T', y='SPEI_Proxy:Q', tooltip=['Combined_Date', 'SPEI_Proxy']
        ).properties(height=400)
        st.altair_chart((spei_raw + danger_lines).interactive(), use_container_width=True)

    with tab4:
        smooth_chart = alt.Chart(df).mark_area(line={'color':'#1e40af'}, color=alt.Gradient(
            gradient='linear', stops=[alt.GradientStop(color='#f87171', offset=0), alt.GradientStop(color='#60a5fa', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )).encode(x='Combined_Date:T', y='SPEI_Rolling:Q').properties(height=400)
        st.altair_chart((smooth_chart + danger_lines).interactive(), use_container_width=True)

    # --- UPDATED AI TAB ---
    with tab5:
        st.subheader("🤖 AI Strategic Briefing")
        if st.button("Analyze with Gemini"):
            if st.session_state.saved_key:
                try:
                    genai.configure(api_key=st.session_state.saved_key)
                    
                    # 1. Dynamically find what your key actually supports
                    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    
                    if not available_models:
                        st.error("Your API key doesn't seem to have access to any Generative Models. Please check your Google AI Studio account.")
                    else:
                        # 2. Pick the best one from YOUR list (prioritizing 1.5 flash if present)
                        best_model = None
                        for pref in ['models/gemini-1.5-flash', 'models/gemini-pro', 'gemini-pro']:
                            if pref in available_models:
                                best_model = pref
                                break
                        
                        if not best_model:
                            best_model = available_models[0] # Use whatever is first

                        model = genai.GenerativeModel(best_model)
                        prompt = (f"Act as a climatologist. Analyzing 1970-2025 data. "
                                  f"Trend: {mk_res.trend}, Slope: {mk_res.slope:.6f}. "
                                  f"Max drought duration: {max_duration} months. "
                                  f"Briefly assess the water security risk.")
                        
                        with st.spinner(f"Analyzing with {best_model}..."):
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI System Error: {e}")
            else:
                st.warning("Enter API Key in sidebar.")

    with tab6:
        st.subheader("🔮 Predictive Modeling (ARIMA 2,0,1)")
        try:
            series = df['SPEI_Proxy'].values
            model = ARIMA(series, order=(2, 0, 1))
            model_fit = model.fit()
            forecast_res = model_fit.get_forecast(steps=12)
            forecast_mean = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int(alpha=0.05) 
            last_date = df['Combined_Date'].max()
            future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=12, freq='MS')
            forecast_df = pd.DataFrame({'Date': future_dates, 'SPEI_Forecast': forecast_mean, 'Lower_CI': conf_int[:, 0], 'Upper_CI': conf_int[:, 1]})
            
            base_f = alt.Chart(forecast_df).encode(x='Date:T')
            band = base_f.mark_area(opacity=0.2, color='orange').encode(y='Lower_CI:Q', y2='Upper_CI:Q')
            line = base_f.mark_line(color='orange', strokeDash=[5,5], point=True).encode(y='SPEI_Forecast:Q')
            st.altair_chart((band + line + danger_lines).properties(height=400).interactive(), use_container_width=True)
        except Exception as e:
            st.error(f"Forecast Error: {e}")
else:
    st.info("👋 Upload data to generate analysis.")