@echo off
REM CAFIN installer for Windows. Double-click this file, or run it from a terminal.
setlocal
cd /d "%~dp0"

REM The pinned Cellpose 3 environment supports Python 3.10 and 3.11 only.
set "PY="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -V >nul 2>&1
    if not errorlevel 1 set "PY=py -3.11"
    if not defined PY (
        py -3.10 -V >nul 2>&1
        if not errorlevel 1 set "PY=py -3.10"
    )
)

if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] <= (3, 11)))" >nul 2>&1
        if not errorlevel 1 set "PY=python"
    )
)

if not defined PY (
    echo Python 3.10 or 3.11 was not found. Install 64-bit Python 3.11 from https://www.python.org/downloads/
    echo Remember to tick "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Installing CAFIN in its own environment. This can take several minutes.
%PY% install.py --launch %*
if errorlevel 1 (
    echo.
    echo The installation did not finish. Scroll up to see the error.
) else (
    echo.
    echo Installation finished. The app should now be open in your browser.
)
pause
