#!/usr/bin/env bash
# Start the CAFIN GUI on macOS or Linux. Double-click on macOS.
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ ! -x ".venv/bin/python" ]; then
    echo "CAFIN has not been installed in this folder yet."
    echo "Run ./install_mac.command once, then open this file again."
    read -r -p "Press Enter to close."
    exit 1
fi

exec ".venv/bin/python" -m streamlit run cafin_gui.py \
    --server.address=127.0.0.1 --server.headless=false
