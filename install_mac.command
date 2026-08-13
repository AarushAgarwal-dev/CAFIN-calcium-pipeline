#!/usr/bin/env bash
# CAFIN installer for macOS and Linux. Double-click on macOS, or run: bash install_mac.command
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# The pinned Cellpose 3 environment supports Python 3.10 and 3.11. Prefer 3.11.
for CANDIDATE in python3.11 python3.10; do
    if command -v "$CANDIDATE" >/dev/null 2>&1; then
        PY="$CANDIDATE"
        break
    fi
done
if [ -z "${PY:-}" ]; then
    echo "Python 3.10 or 3.11 was not found. Install 64-bit Python 3.11:"
    echo "  macOS:  brew install python@3.11   (or download from python.org)"
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
