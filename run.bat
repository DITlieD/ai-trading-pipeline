@echo off
REM Navigate to project folder
cd "C:\Users\LieD\Desktop\Trading pipeline"

REM Activate virtual environment
call venv_trading\Scripts\activate.bat

REM Set Alpaca keys (replace with your actual keys)
set ALPACA_API_KEY=PKNO0VQN4IFGX1U13GRA
set ALPACA_API_SECRET=bSHXghnIxVoQXhEomyohmGGZzFGJWlhR7cNE3MKE

REM Run Streamlit app
streamlit run app.py

pause
