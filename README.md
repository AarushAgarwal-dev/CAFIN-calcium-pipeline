# CAFIN: calcium transient analysis in epithelial cell layers in vivo

CAFIN measures single-cell calcium (Ca2+) dynamics in the zebrafish larval fin epithelium,
before and after cytoskeletal disruption with Latrunculin. It reads two-channel confocal
time-lapse stacks (a membrane channel for structure and a calcium channel for signal) and
produces per-cell ΔF/F0 traces, tissue-level statistics, figures, and a generated draft of the
manuscript. There is also a Streamlit GUI for interactive work, with registration playback,
cell tracking, trace clustering, and optional AI interpretation.

## Quick start

One line downloads the code, creates an isolated `.venv`, installs the GUI, creates a synthetic
demo recording, checks the installation, and opens the app. Python 3.11 is recommended.

**macOS or Linux:**

```bash
git clone https://github.com/AarushAgarwal-dev/CAFIN-calcium-pipeline.git && cd CAFIN-calcium-pipeline && python3 install.py --launch
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/AarushAgarwal-dev/CAFIN-calcium-pipeline.git; cd CAFIN-calcium-pipeline; python install.py --launch
```

You need 64-bit Python 3.10 to 3.12 and git. Python 3.11 has the broadest compatibility with the
scientific and GPU packages. Already have the folder? Run `python install.py --launch`. Next time,
double-click
`run_gui.bat` on Windows or `run_gui.command` on macOS.

## How it works

The pipeline runs in eight steps.

1. Registration. Corrects tissue motion in the time-lapse. Rigid alignment uses OpenCV ECC;
   non-rigid alignment uses itk-elastix B-spline. Registration is computed on the membrane channel
   and the same transform is applied to the calcium channel, so the two stay aligned.
2. Segmentation. Cellpose (cyto3, channels=[2,0], diameter about 15 px) runs on the reference
   membrane frame and returns a labelled cell mask.
3. Background subtraction. Three signal-free regions are sampled in each frame. Values outside
   1.5x IQR are dropped, each region is reduced to its median, and the mean of the three medians
   is subtracted from the frame and clipped at zero.
4. ROI selection (optional). Draw a rectangle; any cell whose mask touches it is kept and exported
   separately.
5. Trace extraction. For each cell, the mean pixel value inside its mask is measured per frame.
   Traces are normalised as ΔF/F0 = (Ft - F0)/F0, where F0 is the mean of the lowest-activity frames.
6. Cell tracking (optional). Every frame is segmented on its own, then cells are linked into stable
   IDs. Matching combines an IoU, size, and centroid score with Hungarian assignment, propagates
   forward and backward from a reference frame, and closes short gaps.
7. Analysis. Per-cell metrics include peak ΔF/F0, transient rate, and per-frame area. Tissue metrics
   include spatial heterogeneity (the coefficient of variation of peak amplitude) and temporal
   synchronization (the mean pairwise correlation). It also computes correlation against inter-cell
   distance and how activity changes over the recording.
8. Statistics. Normality is checked with Shapiro-Wilk. If both groups are normal it uses a Welch
   t-test, otherwise Mann-Whitney U. Effect sizes are reported as the rank-biserial correlation, and
   regional comparisons use Kruskal-Wallis.

## Installing

If you would rather not use a terminal, double-click `install_windows.bat` on Windows or
`install_mac.command` on macOS. Both call the same installer.

`install.py` creates a local `.venv` by default, so it does not alter your normal Python packages.
Use `--launch` to open the GUI, `--cpu` to skip GPU builds, `--with-ai` for Amazon Bedrock, or
`--with-paper` for manuscript generation. `--no-venv` is available for advanced users.

To install the GUI by hand, use `pip install -r requirements-gui.txt`. Optional dependencies are in
`requirements-ai.txt` and `requirements-paper.txt`. The processed-data paper workflow has its own
smaller `REPRODUCE/requirements.txt`.

### Graphics card support

Cellpose segmentation is the slow step, and it is the part that runs on the GPU. Registration
(OpenCV ECC and itk-elastix) is CPU-only either way.

| Hardware | Backend | Installed by |
|---|---|---|
| NVIDIA, Windows or Linux | CUDA | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| AMD or Intel, Windows | DirectML | `pip install torch-directml` |
| AMD, Linux | ROCm | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| Apple Silicon | MPS | stock `pip install torch` |

The installer picks the right one automatically. The GUI shows which backend it found next to the
"Use GPU for segmentation" checkbox. If a GPU backend fails partway through a run, segmentation
falls back to the CPU and continues.

An AMD card in an Intel Mac (a Radeon Pro 5500 XT in a 2019 Mac Pro or 2020 iMac, say) has no
PyTorch GPU backend, since MPS requires Apple Silicon. That run stays on the CPU. The same card in
a Windows machine works through DirectML.

## Using it

### GUI

```bash
streamlit run cafin_gui.py
```

The first launch opens a small synthetic demo, clearly labelled as non-biological. For your data,
choose a trial folder containing `membrane/` and `ca2/` subfolders. Both channels need the same
number of numbered `.tif` frames. Pick a method and click Run analysis. Run `python preflight.py
--data /path/to/trial` for a standalone input check.

The three methods are:

* Rigid (ECC): global motion correction with a fixed frame-0 mask.
* Elastic (itk-elastix): non-rigid B-spline correction, at fast, balanced, or accurate quality.
* Cell tracking: segments every frame and links cells into stable IDs, without registration.

The tabs cover the registration overlay with frame playback, segmentation, ROI drawing, per-cell
traces and a heatmap, PCA plus K-means clustering, tracking, statistics, and CSV or GIF export.

### Reproduce the study

```bash
cd REPRODUCE
python run_all.py
```

This rebuilds the analysis, statistics, figures, draft manuscript (`.docx`), and `PROOF.md` from the
processed per-cell data distributed in the repository. Results are written to `REPRODUCE/results/`.
A fixed random seed makes re-runs identical. More detail is in
[`REPRODUCE/README_REPRODUCE.md`](REPRODUCE/README_REPRODUCE.md).

`python run_all.py --recompute` additionally starts from original raw LATA1 TIFFs. Those large
microscopy files are not distributed on GitHub, so this mode requires the user to supply them in
the documented folder layout.

### AI interpretation (optional)

The Clustering tab can write a short findings narrative from the clustering, using open-source
models on Amazon Bedrock (Llama 3.3 70B by default; Llama 3.1, DeepSeek-R1, and Mixtral are also
available) through the Converse API.

1. Sign in with `aws login` (or `aws configure`).
2. Enable access to your chosen model in the Bedrock console for your region.
3. In the GUI, fill in the background box (drug, concentration, protocol), click Test connection,
   then run either the cluster summary or the full time-course analysis.

The model only sees the numeric summaries shown in the "data sent to the model" panel. Treat its
output as a hypothesis rather than a result.

## Layout

```
cafin_core.py        registration (ECC, itk-elastix), segmentation, background, ΔF/F0, ROI, clustering
cafin_track.py       cell tracking: per-frame segmentation linked by Hungarian matching and gap closing
cafin_pipeline.py    scripted end-to-end pipeline (no GUI)
cafin_gui.py         Streamlit application
cafin_ai.py          Amazon Bedrock (open-source models) narration of clustering results
demo_data.py         creates the synthetic first-run example
preflight.py         checks packages, GPU fallback, and trial-folder structure
requirements-*.txt  GUI plus optional AI and paper dependency groups
REPRODUCE/           scripts, results, figures, and the generated paper
  run_all.py             recompute, analyse, build figures, build paper
  reproduce.py           metrics, statistics, quantitative figures
  figures_paper.py       paper-style figures from real images
  generate_paper.py      builds the manuscript and PROOF.md
  recompute_from_raw.py  regenerates ΔF/F0 from raw TIFFs
  WORKLOG.md             record of the analysis and its caveats
  results/               CSVs, results.json, figures
LATA1TRAIL/, LATA2TRIAL/  per-trial processed data (before and after drug)
```

Raw microscopy stacks (`.tif`, `.nd2`) and other large binaries are not tracked. Only the processed
per-cell CSVs needed to re-run the analysis are included. The tiny TIFF stack under `demo_data/` is
synthetic and is tracked only so a new user can exercise the GUI immediately.

## Data

Each trial folder has a baseline (BEFOREDRUG) and a Latrunculin (AFTERDRUG) condition. The files are:

* `all_cells_raw.csv`: raw per-cell mean intensity per frame
* `all_cells_normalized.csv`: per-cell ΔF/F0
* `roi_cells_normalized.csv`: ΔF/F0 for the ROI cells
* `centroids_0.csv`: cell centroids, used for spatial analysis

To run from raw images, a trial folder needs `membrane/` and `ca2/` subfolders of numbered `.tif`
frames, for example `..._0000.tif`, `..._0001.tif`, and so on.

## Notes

If installation or a dataset fails validation, see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

Per-frame Cellpose segmentation, used by the tracking method, is slow on CPU, roughly 30 to 60
seconds per frame. Use the frame-step control to subsample, or run a CUDA build of torch. The
itk-elastix registration takes about 0.5, 3, or 5 seconds per frame at fast, balanced, or accurate
quality.

On the statistics: the pooled per-cell p-values treat individual cells as replicates. With only two
biological trials per condition, the per-trial direction of the effect is the primary evidence.
This is discussed in the generated paper and in `REPRODUCE/WORKLOG.md`.

## Acknowledgements

The cell-tracking algorithm follows Linlin Li's Cell_Tracking_2D, extended here with Hungarian
assignment and gap closing. Segmentation uses [Cellpose](https://github.com/MouseLand/cellpose).
Non-rigid registration uses [ITKElastix](https://github.com/InsightSoftwareConsortium/ITKElastix).
