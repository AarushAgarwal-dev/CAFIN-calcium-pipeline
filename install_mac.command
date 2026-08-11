#!/bin/bash
# CAFIN installer for macOS and Linux. Double-click on macOS, or run: bash install_mac.command
cd "$(dirname "$0")" || exit 1

if command -v python3.11 >/dev/null 2>&1; then
    PY=python3.11
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "Python was not found. Install Python 3.9 or newer:"
    echo "  macOS:  brew install python   (or download from python.org)"
    echo "  Linux:  sudo apt install python3 python3-pip python3-venv"
    read -r -p "Press Enter to close."
    exit 1
fi

echo "Installing CAFIN in its own environment. This can take several minutes."
"$PY" install.py --launch "$@"
status=$?

echo
if [ $status -eq 0 ]; then
    echo "Installation finished. The app should now be open in your browser."
else
    echo "The installation did not finish. Scroll up to see the error."
fi
read -r -p "Press Enter to close."
exit $status
