@echo off
echo 🌍 Starting Climate Anomaly Dashboard...
cd /d %~dp0

:: Check if venv exists, if not create it
if not exist "venv\" (
    echo 📦 Creating Virtual Environment...
    python -m venv venv
)

:: Activate and Install/Run
echo 🐍 Activating Environment and Launching...
call venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

pause