#!/usr/bin/env python3
"""Safe one-step installer for the CAFIN desktop GUI."""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
REQ_GUI = os.path.join(HERE, "requirements-gui.txt")
CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
ROCM_INDEX = "https://download.pytorch.org/whl/rocm6.2"


def run(cmd, required=True, cwd=HERE):
    print("  $ " + " ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=cwd)
    if required and code:
        raise RuntimeError(f"Command failed with exit code {code}")
    return code


def venv_python():
    return (os.path.join(VENV, "Scripts", "python.exe") if platform.system() == "Windows"
            else os.path.join(VENV, "bin", "python"))


def gpu_vendor():
    try:
        if platform.system() == "Windows":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "(Get-CimInstance Win32_VideoController).Name"]
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).lower()
        elif platform.system() == "Darwin":
            return "apple" if platform.machine() in ("arm64", "aarch64") else "none"
        else:
            out = subprocess.check_output("lspci | grep -Ei 'vga|3d|display'", shell=True,
                                          text=True, stderr=subprocess.DEVNULL).lower()
    except Exception:
        return "none"
    if "nvidia" in out or "geforce" in out or "quadro" in out:
        return "nvidia"
    if "amd" in out or "radeon" in out or "advanced micro" in out:
        return "amd"
    if "intel" in out:
        return "intel"
    return "none"


def install_torch(python, vendor, cpu=False):
    pip = [python, "-m", "pip", "install"]
    if cpu or vendor == "none" or vendor == "apple":
        return run(pip + ["torch"])
    if vendor == "nvidia":
        return run(pip + ["torch", "--index-url", CUDA_INDEX])
    if vendor == "amd" and platform.system() == "Linux":
        return run(pip + ["torch", "--index-url", ROCM_INDEX])
    if vendor in ("amd", "intel") and platform.system() == "Windows":
        # DirectML is optional and version-sensitive. Failure is followed by a tested CPU fallback.
        if run(pip + ["torch-directml"], required=False) == 0:
            return 0
        print("DirectML was unavailable for this Python version. Using CPU PyTorch instead.")
    return run(pip + ["torch"])


def main():
    ap = argparse.ArgumentParser(description="Install CAFIN in an isolated .venv.")
    ap.add_argument("--launch", action="store_true", help="open the GUI after installation")
    ap.add_argument("--cpu", action="store_true", help="skip GPU-specific PyTorch builds")
    ap.add_argument("--with-ai", action="store_true", help="install optional Amazon Bedrock support")
    ap.add_argument("--with-paper", action="store_true", help="install paper-generation packages")
    ap.add_argument("--no-venv", action="store_true",
                    help="advanced: install into the current Python environment")
    # Old documented flag remains accepted; isolation is now already the default.
    ap.add_argument("--venv", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    print(f"CAFIN installer\n  OS: {platform.system()} {platform.release()}\n"
          f"  Bootstrap Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("CAFIN requires Python 3.10 or newer. Python 3.11 is recommended.")
        return 1
    if sys.version_info >= (3, 13):
        print("WARNING: Python 3.13+ is not fully supported by every Cellpose/GPU package.\n"
              "Python 3.11 is strongly recommended. Installation will continue with CPU fallback.")

    python = sys.executable
    if not args.no_venv:
        if not os.path.exists(venv_python()):
            print(f"\nCreating isolated environment: {VENV}")
            run([python, "-m", "venv", VENV])
        python = venv_python()
    print(f"  Installing into: {python}")

    try:
        run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        vendor = "none" if args.cpu else gpu_vendor()
        print(f"  Detected GPU vendor: {vendor}")
        install_torch(python, vendor, args.cpu)
        run([python, "-m", "pip", "install", "-r", REQ_GUI])
        if args.with_ai:
            run([python, "-m", "pip", "install", "-r", "requirements-ai.txt"])
        if args.with_paper:
            run([python, "-m", "pip", "install", "-r", "requirements-paper.txt"])
        run([python, "demo_data.py"])
        run([python, "preflight.py", "--strict"])
    except RuntimeError as exc:
        print(f"\nInstallation did not finish: {exc}\n"
              "See TROUBLESHOOTING.md or rerun with --cpu if a GPU package failed.")
        return 1

    print("\nInstallation complete. Demo data is ready.\n"
          "Start later with run_gui.bat (Windows) or ./run_gui.command (macOS/Linux).")
    if args.launch:
        # headless=false asks Streamlit to open the default browser on a normal desktop.
        # Binding to loopback avoids exposing a local microscopy session to the network.
        return run([python, "-m", "streamlit", "run", "cafin_gui.py",
                    "--server.address=127.0.0.1", "--server.headless=false"], required=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
