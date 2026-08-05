@echo off
REM Start the CAFIN GUI on Windows. Double-click this file.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run cafin_gui.py
) else (
    where py >nul 2>&1 && (py -3 -m streamlit run cafin_gui.py) || (python -m streamlit run cafin_gui.py)
)
pause
