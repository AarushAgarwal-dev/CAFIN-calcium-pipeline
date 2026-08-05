@echo off
REM CAFIN installer for Windows. Double-click this file, or run it from a terminal.
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PY=python"
    ) else (
        echo Python was not found. Install Python 3.9+ from https://www.python.org/downloads/
        echo Remember to tick "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
)

echo Installing CAFIN. This can take several minutes.
%PY% install.py %*
if errorlevel 1 (
    echo.
    echo The installation did not finish. Scroll up to see the error.
) else (
    echo.
    echo Installation finished. Start the app with run_gui.bat
)
pause
