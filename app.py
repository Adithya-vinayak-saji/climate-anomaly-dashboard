import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
from fpdf import FPDF
import io
from PIL import Image

st.set_page_config(
    page_title="Climate Anomaly Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Background Styling ---
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://stock.adobe.com/in/images/global-warming-vs-climate-change-planet-earth-ecology-concept-global-warming-concept-the-effect-of-environment-climate-change/573969248");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .title-text {
            font-size: 3rem;
            font-weight: 700;
            color: white;
            text-shadow: 2px 2px 4px #000000;
        }
        .subtitle-text {
            font-size: 1.3rem;
            color: #f0f0f0;
            text-shadow: 1px 1px 2px #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# --- Header Section ---
col1, col2 = st.columns([1, 4])
with col1:
    try:
        logo = Image.open("Logo App.jpeg")
        st.image(logo, width=100)
    except:
        st.write("")  # Skip logo if not found
with col2:
    st.markdown('<div class="title-text">🌦️ Climate Anomaly Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Visualize seasonal trends, detect anomalies, and generate insights from climate data.</div>', unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload your anomaly dataset (CSV or Excel)", type=["csv", "xlsx"])

uploaded_file = st.file_uploader("📤 Upload your climate data file (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV file loaded successfully.")
    else:
        # Load all sheet names
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names

        # Let user select a sheet
        selected_sheet = st.selectbox("📄 Select a sheet to load", sheet_names)

        # Load the selected sheet
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        st.success(f"✅ Loaded sheet: {selected_sheet}")

# Acceptable time reference columns
time_columns = ["Date", "Months", "DayOfYear"]
available_time_cols = [col for col in time_columns if col in df.columns]

if not available_time_cols:
    st.error("❌ Your file must contain at least one of these columns: 'Date', 'Months', or 'DayOfYear'.")
    st.stop()

# Let user choose which time column to use
selected_time_col = st.selectbox("🕒 Select a time column to use", available_time_cols)
    
    if selected_time_col == "Date":
    try:
        df["Parsed Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
        st.success("✅ Detected format: YYYY-MM.")
    except:
        try:
            df["Parsed Date"] = pd.to_datetime("2022-" + df["Date"].astype(str), format="%Y-%m-%d")
            st.success("✅ Detected format: MM-DD (assumed year 2022).")
        except:
            try:
                df["Parsed Date"] = pd.to_datetime(df["Date"].astype(int), format="%m")
                st.success("✅ Detected numeric month format (1–12).")
            except:
                st.error("❌ Could not parse 'Date'.")
                st.stop()
    df["Month"] = df["Parsed Date"].dt.month
    df["Months"] = df["Parsed Date"].dt.strftime("%B")
    df["DayOfYear"] = df["Parsed Date"].dt.dayofyear

elif selected_time_col == "Months":
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["Month"] = df["Months"].map(month_map)
    df["Parsed Date"] = pd.to_datetime(df["Month"], format="%m")
    df["DayOfYear"] = df["Parsed Date"].dt.dayofyear

elif selected_time_col == "DayOfYear":
    df["DayOfYear"] = df["DayOfYear"].astype(int)
    df["Parsed Date"] = pd.to_datetime("2022-01-01") + pd.to_timedelta(df["DayOfYear"] - 1, unit="D")
    df["Month"] = df["Parsed Date"].dt.month
    df["Months"] = df["Parsed Date"].dt.strftime("%B")

    # Extract month and year
    df["Month"] = df["Parsed Date"].dt.month
    df["Year"] = df["Parsed Date"].dt.year

    # Year filter
    min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
    year_range = st.slider("📅 Select year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

    # Season filter
    season_map = {
        "Winter (Dec-Feb)": [12, 1, 2],
        "Summer (Mar-May)": [3, 4, 5],
        "Monsoon (Jun-Sep)": [6, 7, 8, 9],
        "Post-Monsoon (Oct-Nov)": [10, 11]
    }
    season_choice = st.selectbox("🌦️ Select season (optional)", ["All"] + list(season_map.keys()))
    if season_choice != "All":
        df = df[df["Month"].isin(season_map[season_choice])]

    # Panel 1: Time Series
    st.header("📈 Panel 1: Time Series Plot")
    selected_params = st.multiselect("Select variables to plot", df.columns[1:], default=df.columns[1])
    if selected_params:
        st.line_chart(df.set_index("Date")[selected_params])

    # Panel 2: Boxplot
    st.header("📊 Panel 2: Boxplot")
    box_param = st.selectbox("Select variable for boxplot", df.columns[1:])
    fig1, ax1 = plt.subplots()
    sns.boxplot(y=df[box_param], ax=ax1)
    ax1.set_title(f"Boxplot of {box_param}")
    st.pyplot(fig1)

    # Panel 3: Histogram
    st.header("📉 Panel 3: Histogram")
    hist_param = st.selectbox("Select variable for histogram", df.columns[1:], key="hist")
    fig2, ax2 = plt.subplots()
    ax2.hist(df[hist_param].dropna(), bins=20, color="skyblue", edgecolor="black")
    ax2.set_title(f"Histogram of {hist_param}")
    st.pyplot(fig2)

    # Panel 4: Correlation Heatmap
    st.header("🧊 Panel 4: Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", ax=ax3)
    ax3.set_title("Correlation Heatmap")
    st.pyplot(fig3)

    # Panel 5: Trend Detection
    st.header("📉 Panel 5: Trend Detection with Kendall Tests")
    freq_choice = st.radio("Select data frequency:", ["Monthly", "Daily"])
    trend_param = st.selectbox("Select variable for trend analysis", df.columns[1:], key="trend")

    data_series = df[trend_param].dropna()
    if freq_choice == "Monthly":
        result = mk.seasonal_test(data_series, period=12)
    else:
        result = mk.hamed_rao_modification_test(data_series)

    # Auto-generated insight
    insight = f"The variable **{trend_param}** shows a **{result.trend.lower()} trend**"
    if result.h:
        insight += f" that is **statistically significant** (p = {result.p:.4f})."
    else:
        insight += f", but it is **not statistically significant** (p = {result.p:.4f})."
    insight += f" The estimated rate of change is **{result.slope:.4f}** per time unit."

    st.markdown(insight)

    # Generate trend chart
    fig4, ax4 = plt.subplots()
    x = np.arange(len(df))
    y = df[trend_param].values
    ax4.plot(df["Date"], y, label="Anomaly", color="blue")
    slope, intercept = np.polyfit(x, y, 1)
    ax4.plot(df["Date"], slope * x + intercept, linestyle="--", color="red", label="Trend")
    ax4.set_title(f"{trend_param} Trend")
    ax4.set_ylabel(trend_param)
    ax4.legend()
    st.pyplot(fig4)

    # Save charts to buffers
    def save_chart(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    chart_buffers = {
        "Time Series": save_chart(fig4),
        "Histogram": save_chart(fig2),
        "Boxplot": save_chart(fig1)
    }

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Climate Anomaly Trend Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"""
Variable: {trend_param}
Season: {season_choice}
Years: {year_range[0]} – {year_range[1]}
Test: {'Seasonal Kendall' if freq_choice == 'Monthly' else 'Mann-Kendall (Hamed-Rao)'}
Trend: {result.trend}
Significant: {'Yes' if result.h else 'No'}
p-value: {result.p:.4f}
Z-score: {result.z:.2f}
Slope: {result.slope:.4f}
""")

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Insight", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, insight)

    for title, buf in chart_buffers.items():
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.image(buf, x=10, y=30, w=180)

    # Export PDF
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    st.download_button(
        label="📄 Download Full Report (PDF)",
        data=pdf_buffer,
        file_name=f"{trend_param}_trend_report.pdf",
        mime="application/pdf"
    )

else:
    st.warning("👆 Please upload a dataset to begin.")