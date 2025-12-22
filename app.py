import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
import io
from PIL import Image

# --- Page Config ---
st.set_page_config(
    page_title="Climate Anomaly Dashboard",
    page_icon="🌍",
    layout="wide"
)

# --- High-Contrast Dark Mode Styling ---
def set_styling():
    st.markdown(
        """
        <style>
        /* 1. Main Dashboard Background */
        .stApp {
            background: linear-gradient(to bottom, #0f0c29, #302b63, #24243e);
            color: #ffffff;
        }

        /* 2. Sidebar styling: Dark with Cyan border */
        [data-testid="stSidebar"] {
            background-color: #1a1a2e !important;
            border-right: 2px solid #00d2ff;
        }

        /* 3. FIX: Force ALL Sidebar text and labels to White */
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* 4. FIX: Remove the white area from the File Uploader Box */
        [data-testid="stFileUploadDropzone"] {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 2px dashed #00d2ff !important;
            color: #ffffff !important;
        }
        
        /* 5. FIX: Browse Files Button visibility (Bright Cyan with Dark Text) */
        button[kind="secondary"] {
            color: #1a1a2e !important;
            background-color: #00d2ff !important;
            border: 2px solid #00d2ff !important;
            font-weight: bold !important;
        }
        
        button[kind="secondary"]:hover {
            background-color: #ffffff !important;
            color: #1a1a2e !important;
        }

        /* Header Styling */
        .title-text {
            font-size: 3rem;
            font-weight: 800;
            color: #00d2ff;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        }
        hr { border: 0.5px solid #00d2ff; }
        </style>
        """,
        unsafe_allow_html=True
    )

set_styling()

# --- Header ---
st.markdown('<div class="title-text">🌍 Climate Anomaly Dashboard</div>', unsafe_allow_html=True)
st.markdown("Analyze seasonal trends and inspect specific extreme weather anomalies.")
st.markdown("---")

# --- Sidebar: Data Management ---
st.sidebar.markdown("### 📂 Data Management")
dataset_type = st.sidebar.radio("Dataset Category", ["Climate Data", "Anomaly Data"], key="ds_type")
uploaded_file = st.sidebar.file_uploader(f"Upload {dataset_type}", type=["csv", "xlsx"])

df = None

if uploaded_file:
    # Handle Excel Sheets
    if uploaded_file.name.endswith(".xlsx"):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        selected_sheet = st.sidebar.selectbox("📄 Select a Sheet", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    else:
        df = pd.read_csv(uploaded_file)

# --- Analysis Logic ---
if df is not None:
    # 1. Flexible Column Detection
    cols_lower = [c.lower() for c in df.columns]
    target_col = None
    
    if "date" in cols_lower:
        target_col = df.columns[cols_lower.index("date")]
        df["Parsed Date"] = pd.to_datetime(df[target_col], errors="coerce")
    elif "months" in cols_lower:
        target_col = df.columns[cols_lower.index("months")]
        df["Parsed Date"] = pd.to_datetime(df[target_col], format="%B", errors="coerce")
    elif "dayofyear" in cols_lower:
        target_col = df.columns[cols_lower.index("dayofyear")]
        df["Parsed Date"] = pd.to_datetime(df[target_col], format="%j", errors="coerce")

    if "Parsed Date" not in df.columns or df["Parsed Date"].isnull().all():
        st.error("❌ Valid time column not detected in this sheet.")
        st.info(f"Detected Columns: {list(df.columns)}")
        st.stop()

    # Clean data and Extract features
    df = df.dropna(subset=["Parsed Date"])
    df["Year"] = df["Parsed Date"].dt.year
    df["Month"] = df["Parsed Date"].dt.month

    # 2. Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Filters")
    
    available_years = sorted(df["Year"].unique().astype(int))
    
    # Check session state for persistence
    if "yr_range" not in st.session_state:
        st.session_state.yr_range = (min(available_years), max(available_years))

    if st.sidebar.button("🔄 Reset Filters"):
        st.session_state.yr_range = (min(available_years), max(available_years))
        st.session_state.sn_choice = "All"
        st.rerun()

    year_filter = st.sidebar.slider("Year Range", min(available_years), max(available_years), 
                                    st.session_state.yr_range, key="yr_range")

    season_map = {
        "Winter (Dec-Feb)": [12, 1, 2], "Summer (Mar-May)": [3, 4, 5],
        "Monsoon (Jun-Sep)": [6, 7, 8, 9], "Post-Monsoon (Oct-Nov)": [10, 11]
    }
    season_choice = st.sidebar.selectbox("Season", ["All"] + list(season_map.keys()), key="sn_choice")

    # Filter Application
    f_df = df[(df["Year"] >= year_filter[0]) & (df["Year"] <= year_filter[1])].copy()
    if season_choice != "All":
        f_df = f_df[f_df["Month"].isin(season_map[season_choice])]

    # 3. Visualizations
    numeric_cols = [c for c in f_df.select_dtypes(include=[np.number]).columns if c not in ["Year", "Month"]]

    if numeric_cols:
        st.subheader("📈 Time Series & Anomaly Detection")
        ts_var = st.selectbox("Select Climate Variable", numeric_cols)
        
        # Anomaly Logic: Mean +/- 2 Standard Deviations
        mean_val = f_df[ts_var].mean()
        std_val = f_df[ts_var].std()
        f_df['is_anomaly'] = (f_df[ts_var] > mean_val + 2*std_val) | (f_df[ts_var] < mean_val - 2*std_val)
        anomalies = f_df[f_df['is_anomaly'] == True]

        fig_ts, ax_ts = plt.subplots(figsize=(12, 5), facecolor='#1e1e2f')
        ax_ts.set_facecolor('#1e1e2f')
        ax_ts.plot(f_df["Parsed Date"], f_df[ts_var], color="#00d2ff", label="Normal Data", alpha=0.7)
        ax_ts.scatter(anomalies["Parsed Date"], anomalies[ts_var], color="#ff4b4b", label="Anomaly Detected", zorder=5)
        
        ax_ts.tick_params(colors='white')
        ax_ts.legend(facecolor='#1e1e2f', labelcolor='white')
        st.pyplot(fig_ts)

        # Summary Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Selected Average", f"{mean_val:.2f}")
        m2.metric("Total Anomalies Found", len(anomalies))
        m3.metric("Standard Deviation (σ)", f"{std_val:.2f}")

        # 4. New: Anomaly Data Preview
        st.markdown("---")
        st.subheader("🔍 Anomaly Data Preview")
        if not anomalies.empty:
            st.write(f"The following table shows data points exceeding the $\pm 2\sigma$ threshold for **{ts_var}**.")
            # Displaying only relevant columns for the table
            display_cols = ["Parsed Date", ts_var, "Year", "Month"]
            st.dataframe(anomalies[display_cols].sort_values("Parsed Date"), use_container_width=True)
        else:
            st.info("No anomalies detected in the current filtered range.")

    else:
        st.warning("No numeric columns available in the selected sheet.")
else:
    st.info("👋 Use the sidebar to upload a file and select a sheet.")