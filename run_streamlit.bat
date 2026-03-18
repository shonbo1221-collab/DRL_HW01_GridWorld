@echo off
echo =========================================
echo  HW1 GridWorld Streamlit Launcher
echo =========================================
echo.
echo Activating Conda environment 'rl_hw1'...
call conda activate rl_hw1
echo Starting Streamlit app...
streamlit run streamlit_app.py
pause
