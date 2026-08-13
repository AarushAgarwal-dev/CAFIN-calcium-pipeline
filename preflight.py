#!/usr/bin/env python3
"""CAFIN installation and input-data diagnostic."""
from __future__ import annotations
import argparse
import importlib
import os
from pathlib import Path
import re
import sys

CORE = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "matplotlib": "matplotlib", "opencv": "cv2", "scikit-image": "skimage",
    "tifffile": "tifffile", "Pillow": "PIL", "Streamlit": "streamlit",
    "Plotly": "plotly", "scikit-learn": "sklearn", "Cellpose": "cellpose",
}
OPTIONAL = {"Elastic registration": "itk", "Amazon Bedrock": "boto3",
            "Paper generation": "docx"}


def validate_data(path):
    errors = []
    root = Path(path)
    for channel in ("membrane", "ca2"):
        folder = root / channel
        if not folder.is_dir():
            errors.append(f"missing {channel}/ folder")
            continue
        files = sorted(folder.glob("*.tif"))
        if not files:
            errors.append(f"no .tif frames in {channel}/")
        elif not any(re.search(r"\d{4}\.tif$", f.name, re.I) for f in files):
            errors.append(f"{channel}/ filenames must end in four digits, e.g. _0000.tif")
    if not errors:
        nm = len(list((root / "membrane").glob("*.tif")))
        nc = len(list((root / "ca2").glob("*.tif")))
        if nm != nc:
            errors.append(f"channel frame counts differ: membrane={nm}, ca2={nc}")
    return errors


def main():
    ap = argparse.ArgumentParser(description="Check CAFIN before starting an analysis.")
    ap.add_argument("--data", help="trial folder containing membrane/ and ca2/")
    ap.add_argument("--strict", action="store_true", help="fail when a core package is missing")
    args = ap.parse_args()
    supported_python = (3, 10) <= sys.version_info[:2] <= (3, 11)
    print(f"Python {sys.version.split()[0]}: "
          f"{'supported by the pinned Cellpose 3 stack' if supported_python else 'unsupported; install Python 3.11'}")
    missing = []
    for label, module in CORE.items():
        try:
            obj = importlib.import_module(module)
            print(f"[OK] {label} {getattr(obj, '__version__', '')}".rstrip())
        except Exception as exc:
            print(f"[MISSING] {label}: {exc}")
            missing.append(label)
    for label, module in OPTIONAL.items():
        try:
            importlib.import_module(module)
            print(f"[OPTIONAL OK] {label}")
        except Exception:
            print(f"[OPTIONAL] {label} is not installed")
    try:
        import cafin_core as cc
        ok, msg, backend = cc.gpu_status()
        print(f"[GPU] {backend}: {msg.splitlines()[0]}")
        if not ok:
            print("      CPU fallback is available; segmentation will be slower.")
    except Exception as exc:
        print(f"[GPU] could not inspect backend: {exc}")
    if args.data:
        problems = validate_data(args.data)
        if problems:
            for problem in problems:
                print(f"[DATA ERROR] {problem}")
            return 2
        print(f"[DATA OK] {os.path.abspath(args.data)}")
    return 1 if (args.strict and missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
