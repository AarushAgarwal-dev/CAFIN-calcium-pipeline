#!/usr/bin/env bash
# =============================================================================
# setup_vastai.sh  --  one-shot setup + launch for a rented vast.ai GPU box
# =============================================================================
# Run it from the folder that contains cafin_gui.py (e.g. /workspace):
#
#     bash setup_vastai.sh              # install deps + launch the GUI on :8501
#     bash setup_vastai.sh --batch      # install deps + run run_all_conditions.py
#     PORT=8600 bash setup_vastai.sh    # use a different Streamlit port
#
# Then, on YOUR local machine, open the SSH tunnel and browse to the GUI:
#     ssh -p <PORT> root@<HOST> -L 8501:localhost:8501
#     # open http://localhost:8501 in your local browser
# =============================================================================
set -euo pipefail

PORT="${PORT:-8501}"
MODE="${1:-gui}"

echo "==> CAFIN vast.ai setup starting"

# --- 1. Python deps (torch+CUDA is preinstalled in the PyTorch template) ------
echo "==> Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    streamlit cellpose opencv-python-headless scikit-image tifffile imageio \
    pillow pandas scipy natsort matplotlib SimpleITK

# --- 2. If torch has no CUDA, install a CUDA build ----------------------------
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "==> No CUDA torch detected -> installing CUDA 12.4 build..."
    pip install --quiet --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124 || \
        echo "!! CUDA torch install failed; will run on CPU."
fi

# --- 3. Report the GPU --------------------------------------------------------
echo "==> GPU status:"
python - <<'PY'
import torch
if torch.cuda.is_available():
    print("   CUDA available:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
else:
    print("   CUDA NOT available (running on CPU) | torch", torch.__version__)
PY

# --- 4. Warm the Cellpose model (downloads cpsam weights once) ----------------
echo "==> Downloading Cellpose model weights (first run only)..."
python - <<'PY' || true
from cellpose import models
models.CellposeModel(gpu=__import__("torch").cuda.is_available())
print("   Cellpose model ready.")
PY

# --- 5. Launch ----------------------------------------------------------------
if [ "$MODE" = "--batch" ] || [ "$MODE" = "batch" ]; then
    echo "==> Running batch pipeline (run_all_conditions.py)..."
    python run_all_conditions.py
    echo "==> Done. See *_output/ folders and results_summary.json"
else
    echo "==> Launching Streamlit GUI on port ${PORT}"
    echo "    On your LOCAL machine run:  ssh -p <PORT> root@<HOST> -L ${PORT}:localhost:${PORT}"
    echo "    then open  http://localhost:${PORT}"
    echo "    (tip: run this inside 'tmux' so it survives a dropped SSH connection)"
    exec streamlit run cafin_gui.py --server.port "${PORT}" --server.headless true \
         --browser.gatherUsageStats false
fi
