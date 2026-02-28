import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import google.generativeai as genai
import pymannkendall as mk
from prophet import Prophet  # New Engine

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
    
    if st.button("🔍 Check API Connection"):
        if input_key:
            try:
                genai.configure(api_key=input_key)
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                st.success(f"Connected! Available: {len(models)} models")
            except Exception as e:
                st.error(f"Failed: {e}")
    
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
    year_col = st.sidebar.selectbox("Year", all_cols, index=0)
    month_col = st.sidebar.selectbox("Month", all_cols, index=1)
    p_col = st.sidebar.selectbox("Precipitation", all_cols, index=4)
    t_col = st.sidebar.selectbox("Temperature", all_cols, index=2)
    w_col = st.sidebar.selectbox("Wind Speed", all_cols, index=5)
    h_col = st.sidebar.selectbox("Humidity", all_cols, index=3)

    # --- DATA PROCESSING ---
    df = df_raw.copy()
    df['Combined_Date'] = pd.to_datetime(df[year_col].astype(str) + '-' + df[month_col].astype(str) + '-01')
    for col in [p_col, t_col, w_col, h_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Combined_Date', p_col, t_col]).sort_values('Combined_Date')

    # --- SPEI CALCULATIONS ---
    df['PET'] = df.apply(lambda r: calculate_simplified_pet(r[t_col], r[w_col], r[h_col]), axis=1)
    df['D'] = df[p_col] - df['PET']
    df['SPEI_Proxy'] = (df['D'] - df['D'].mean()) / df['D'].std()
    df['SPEI_Rolling'] = df['SPEI_Proxy'].rolling(window=12, center=True).mean()
    
    extreme_count = len(df[df['SPEI_Proxy'] <= -2.0])
    max_duration = get_longest_duration(df['SPEI_Proxy'])
    mk_res = mk.original_test(df['SPEI_Proxy'])

    st.title(f"🔍 SPEI Intelligence: {selected_sheet}")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Trend", mk_res.trend.upper())
    with m2: st.metric("Sen's Slope", f"{mk_res.slope:.6f}") 
    with m3: st.metric("Extreme Events", extreme_count)
    with m4: st.metric("Max Duration", f"{max_duration} Months")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📉 Timeline", "⚖️ Water Balance", "🚨 Raw SPEI", "🌊 Smooth Trend", "🤖 AI Insight", "🔮 Prophet Forecast"
    ])

    danger_lines = alt.Chart(pd.DataFrame({'y': [-1.5, -2.0], 'label': ['Severe', 'Extreme']})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q', tooltip='label:N')

    with tab1:
        st.altair_chart(alt.Chart(df).mark_line().encode(x='Combined_Date:T', y=f'{p_col}:Q', color=alt.value('#3b82f6')).properties(height=400).interactive(), use_container_width=True)

    with tab2:
        st.altair_chart(alt.Chart(df).mark_bar().encode(x='Combined_Date:T', y='D:Q', color=alt.condition(alt.datum.D > 0, alt.value('#60a5fa'), alt.value('#f87171'))).properties(height=400).interactive(), use_container_width=True)

    with tab3:
        st.altair_chart((alt.Chart(df).mark_line(point=True).encode(x='Combined_Date:T', y='SPEI_Proxy:Q') + danger_lines).properties(height=400).interactive(), use_container_width=True)

    with tab4:
        st.altair_chart((alt.Chart(df).mark_area(line={'color':'#1e40af'}).encode(x='Combined_Date:T', y='SPEI_Rolling:Q') + danger_lines).properties(height=400).interactive(), use_container_width=True)

    with tab5:
        st.subheader("🤖 AI Strategic Briefing")
        if st.button("Analyze with Gemini"):
            if st.session_state.saved_key:
                try:
                    genai.configure(api_key=st.session_state.saved_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"Climatologist view: Trend={mk_res.trend}, Slope={mk_res.slope:.6f}, Max Drought={max_duration} months. Briefly assess risk."
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")

    with tab6:
        st.subheader("🔮 12-Month Predictive Modeling (Prophet)")
        try:
            # 1. Prepare data for Prophet (requires 'ds' and 'y' columns)
            p_df = df[['Combined_Date', 'SPEI_Proxy']].rename(columns={'Combined_Date': 'ds', 'SPEI_Proxy': 'y'})
            
            # 2. Initialize and fit
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(p_df)
            
            # 3. Predict future
            future = m.make_future_dataframe(periods=12, freq='MS')
            forecast = m.predict(future)
            
            # 4. Filter to show only the forecast period for clarity
            forecast_results = forecast.tail(12)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            # 5. Visualize with Altair
            base_f = alt.Chart(forecast_results).encode(x=alt.X('ds:T', title='Future Timeline'))
            band = base_f.mark_area(opacity=0.3, color='#8b5cf6').encode(y=alt.Y('yhat_lower:Q', title='Predicted SPEI'), y2='yhat_upper:Q')
            line = base_f.mark_line(color='#8b5cf6', strokeDash=[5,5], point=True).encode(y='yhat:Q', tooltip=['ds', 'yhat'])
            
            st.altair_chart((band + line + danger_lines).properties(height=400).interactive(), use_container_width=True)
            st.success("Prophet has successfully detected the seasonal patterns in your data and projected them forward.")
            
        except Exception as e:
            st.error(f"Prophet Error: {e}")
else:
    st.info("👋 Upload climate data to begin.")