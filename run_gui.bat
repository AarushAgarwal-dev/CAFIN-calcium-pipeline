@echo off
REM Start the CAFIN GUI on Windows. Double-click this file.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run cafin_gui.py --server.address=127.0.0.1 --server.headless=false
) else (
    echo CAFIN has not been installed in this folder yet.
    echo Double-click install_windows.bat once, then open this file again.
    pause
    exit /b 1
)
pause
