#!/bin/bash
# Start the CAFIN GUI on macOS or Linux. Double-click on macOS.
cd "$(dirname "$0")" || exit 1
if [ -x ".venv/bin/python" ]; then
    ".venv/bin/python" -m streamlit run cafin_gui.py
elif command -v python3 >/dev/null 2>&1; then
    python3 -m streamlit run cafin_gui.py
else
    python -m streamlit run cafin_gui.py
fi
