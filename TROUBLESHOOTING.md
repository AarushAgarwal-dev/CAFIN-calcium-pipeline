# CAFIN troubleshooting

## Recommended setup

Use 64-bit Python 3.11. The installer creates `.venv`, so CAFIN does not change packages in your
normal Python installation.

```powershell
python install.py --launch
```

Run `\.venv\Scripts\python preflight.py --data "C:\path\to\trial"` on Windows, or
`.venv/bin/python preflight.py --data /path/to/trial` on macOS/Linux, to diagnose an installation
or dataset before analysis.

## The GUI opens but Run analysis is disabled

Choose a folder containing `membrane/` and `ca2/`. Both must contain the same number of TIFF files,
and filenames must end with four digits such as `sample_0000.tif`. Click **Demo** to verify that the
application works without supplying biological data.

## GPU is unavailable

CAFIN automatically runs Cellpose on the CPU when a GPU backend is unavailable or fails. This is
correct but slower. NVIDIA uses CUDA, AMD on Linux uses ROCm, Apple Silicon uses MPS, and AMD or
Intel on Windows may use DirectML. DirectML availability depends on its PyTorch compatibility, so
CPU fallback is expected on some Windows systems.

Rerun `python install.py --cpu` if GPU package installation causes trouble.

## Cellpose does not run on this computer

Cellpose is optional. Install and launch the mask-based GUI without PyTorch or Cellpose using:

```powershell
python install.py --without-cellpose --launch
```

For rigid or elastic analysis, select **Load existing mask** under **Segmentation source**. Upload a
2-D TIFF, PNG, or NPY label image aligned to membrane frame 0. Background must be 0; cells may be a
binary foreground or separate positive integer IDs. CAFIN cleans and relabels it, then performs
registration, trace extraction, normalization, and statistics without running Cellpose.

To clean or convert a mask without any calcium analysis, open **Standalone mask cleaning**. This
tool removes small components, fills holes per cell, optionally removes border cells, splits
disconnected regions sharing an ID, and downloads a standard labelled TIFF.

Cell tracking still requires a segmentation for every frame, so a single loaded mask cannot replace
Cellpose in tracking mode.

## Elastic registration is unavailable

Install the GUI requirement group inside the CAFIN environment:

```powershell
.venv\Scripts\python -m pip install -r requirements-gui.txt
```

Rigid ECC registration remains available if ITK-Elastix cannot run.

## Amazon Bedrock is unavailable

Bedrock is optional. Install it with `python install.py --with-ai`, configure AWS credentials, enable
an offered model in your selected region, and test the connection in the Clustering tab.

## Paper reproduction and raw data

`python REPRODUCE/run_all.py` rebuilds statistics, figures, and the paper from included processed
data. `python REPRODUCE/run_all.py --recompute` additionally requires the original raw LATA1 TIFF
folders. Those large microscopy images are not distributed through GitHub.
