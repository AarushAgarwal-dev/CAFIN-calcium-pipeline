#!/usr/bin/env python3
"""
CAFIN one-step installer.

Works the same on Windows, macOS and Linux: it detects the operating system and
the GPU, installs the matching PyTorch build, then installs everything else.

    python install.py              install into the current Python environment
    python install.py --venv       create ./.venv first and install into that
    python install.py --cpu        skip GPU detection, install the CPU build
    python install.py --launch     start the GUI when the install finishes

GPU handling:
    NVIDIA (Windows/Linux)  CUDA build of torch
    AMD / Intel (Windows)   torch-directml
    AMD (Linux)             ROCm build of torch
    Apple Silicon (macOS)   stock torch, which includes MPS
    anything else           CPU build
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQS = os.path.join(HERE, "REPRODUCE", "requirements.txt")

CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
ROCM_INDEX = "https://download.pytorch.org/whl/rocm6.2"


def say(msg):
    print(f"\n=== {msg}", flush=True)


def run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def gpu_vendor():
    """Return 'nvidia' | 'amd' | 'intel' | 'apple' | 'none' using OS tools only
    (no Python packages needed, since nothing is installed yet)."""
    sysname = platform.system()
    try:
        if sysname == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                text=True, stderr=subprocess.DEVNULL, timeout=30).lower()
        elif sysname == "Darwin":
            if platform.machine() in ("arm64", "aarch64"):
                return "apple"
            out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"],
                                          text=True, stderr=subprocess.DEVNULL,
                                          timeout=60).lower()
        else:
            out = subprocess.check_output("lspci | grep -i 'vga\\|3d\\|display'", shell=True,
                                          text=True, stderr=subprocess.DEVNULL,
                                          timeout=30).lower()
    except Exception:
        return "none"
    if "nvidia" in out or "geforce" in out or "quadro" in out:
        return "nvidia"
    if "amd" in out or "radeon" in out or "advanced micro" in out:
        return "amd"
    if "intel" in out:
        return "intel"
    return "none"


def torch_command(pip, vendor, force_cpu):
    """Return (pip command, human-readable description) for installing torch."""
    sysname = platform.system()
    if force_cpu:
        return pip + ["install", "torch"], "CPU build"
    if vendor == "nvidia":
        return pip + ["install", "torch", "--index-url", CUDA_INDEX], "NVIDIA CUDA build"
    if vendor == "apple":
        return pip + ["install", "torch"], "Apple Silicon build (MPS)"
    if vendor in ("amd", "intel") and sysname == "Windows":
        return pip + ["install", "torch-directml"], f"DirectML build for {vendor.upper()} on Windows"
    if vendor == "amd" and sysname == "Linux":
        return pip + ["install", "torch", "--index-url", ROCM_INDEX], "AMD ROCm build"
    return pip + ["install", "torch"], "CPU build (no supported GPU detected)"


def main():
    ap = argparse.ArgumentParser(description="Install CAFIN and its dependencies.")
    ap.add_argument("--venv", action="store_true", help="create ./.venv and install into it")
    ap.add_argument("--cpu", action="store_true", help="force the CPU build of torch")
    ap.add_argument("--launch", action="store_true", help="start the GUI after installing")
    args = ap.parse_args()

    print("CAFIN installer")
    print(f"  OS      : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python  : {sys.version.split()[0]}")

    if sys.version_info < (3, 9):
        print("\nPython 3.9 or newer is required. Please upgrade Python and run this again.")
        return 1

    python = sys.executable
    if args.venv:
        venv_dir = os.path.join(HERE, ".venv")
        if not os.path.isdir(venv_dir):
            say(f"Creating virtual environment in {venv_dir}")
            if run([python, "-m", "venv", venv_dir]) != 0:
                print("Could not create the virtual environment.")
                return 1
        python = (os.path.join(venv_dir, "Scripts", "python.exe") if platform.system() == "Windows"
                  else os.path.join(venv_dir, "bin", "python"))
        print(f"  Using   : {python}")

    pip = [python, "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip"])

    vendor = "none" if args.cpu else gpu_vendor()
    print(f"  GPU     : {vendor}")

    cmd, desc = torch_command(pip, vendor, args.cpu)
    say(f"Installing PyTorch: {desc}")
    if run(cmd) != 0:
        print("\nThe GPU build failed to install; falling back to the CPU build.")
        run(pip + ["install", "torch"])

    say("Installing the remaining dependencies")
    if not os.path.exists(REQS):
        print(f"Could not find {REQS}")
        return 1
    if run(pip + ["install", "-r", REQS]) != 0:
        print("\nSome dependencies failed to install. Scroll up for the error.")
        return 1

    say("Checking the installation")
    check = (
        "import cafin_core as cc;"
        "ok, msg, be = cc.gpu_status();"
        "print('  GPU backend :', be if ok else 'cpu');"
        "print('  ' + msg.splitlines()[0])"
    )
    subprocess.call([python, "-c", check], cwd=HERE)

    print("\nDone. Start the GUI with:")
    if args.venv:
        vpy = ".venv\\Scripts\\python" if platform.system() == "Windows" else ".venv/bin/python"
        print("  " + vpy + " -m streamlit run cafin_gui.py")
    else:
        print("  streamlit run cafin_gui.py")

    if args.launch:
        say("Starting the GUI")
        subprocess.call([python, "-m", "streamlit", "run", "cafin_gui.py"], cwd=HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
