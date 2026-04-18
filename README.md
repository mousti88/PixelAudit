# PixelAudit

PixelAudit is a transparent AI image forensics project using a C++ OpenCV core and a simple Streamlit UI.

## Current scope (Iteration 3+)
- C++20 + OpenCV core detector.
- Implemented tests:
  - Luminance-Gradient PCA Analysis (classical CV).
  - Frequency Spectrum and Ringing Analysis (classical CV).
  - JPEG and Compression Forensics (classical CV).
  - Optical/Photometric Plausibility Checks (classical CV).
  - Sensor Noise and PRNU/CFA Consistency Checks (classical CV).
  - Vanishing Point Geometry Consistency (classical CV).
- Optional non-classical add-on:
  - Latent-manifold module (currently explicit placeholder).
- JSON report with per-test metrics and artifact image paths.
- Streamlit UI with two-column layout and collapsible test sections.

## Repository Layout
- `include/` C++ headers
- `src/` C++ sources
- `ui/` Streamlit app
- `scripts/` dataset export and evaluation helpers
- `tests/` C++ smoke test

## macOS Setup (Homebrew)
1. Install dependencies:
```bash
brew install cmake ninja opencv
```
2. Configure and build:
```bash
cmake --preset default
cmake --build --preset default
```
3. Run CLI:
```bash
./build/pixelaudit_cli --input /path/to/image.jpg --output-dir ./reports/run1
```
Optional calibrated fusion:
```bash
./build/pixelaudit_cli --input /path/to/image.jpg --output-dir ./reports/run1 --calibration-file ./reports/calibration/pixelaudit_calibration.json
```
4. Inspect generated files:
- `./reports/run1/report.json`
- `./reports/run1/artifacts/*.png`

## Streamlit UI
1. Create environment and install UI deps:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ui/requirements.txt
```
2. Run:
```bash
streamlit run ui/app.py
```
3. If CLI binary is in a non-default location:
```bash
export PIXELAUDIT_CLI=/absolute/path/to/pixelaudit_cli
```

## Dataset Helper (Parveshiiii/AI-vs-Real)
1. Install script deps:
```bash
pip install -r scripts/requirements.txt
```
2. Export a subset:
```bash
python scripts/export_hf_dataset.py --output data/ai_vs_real --split train --max-per-class 80 --ai-label-value 0
```
Use `--ai-label-value` to match dataset labeling conventions and avoid class inversion.
3. Evaluate:
```bash
python scripts/evaluate_dataset.py --data-root data/ai_vs_real --cli build/pixelaudit_cli --limit-per-class 100
```

4. Calibrate fusion weights and bias:
```bash
python scripts/calibrate_dataset.py --data-root data/ai_vs_real --cli build/pixelaudit_cli --limit-per-class 120 --output reports/calibration/pixelaudit_calibration.json
```

5. Run with calibrated fusion weights:
```bash
./build/pixelaudit_cli --input /path/to/image.jpg --output-dir ./reports/run_calibrated --calibration-file reports/calibration/pixelaudit_calibration.json
```

## Notes on Transparency
- Each test reports: score, status, confidence, explanation, raw metrics, and artifacts.
- Final probability is a weighted fusion of all test scores.
- Contribution map shows how much each test influenced the final output.

## Test Explanations (What is Used, What it Shows)
### 1) Luminance-Gradient PCA Analysis (Classical)
What is used:
- RGB to luminance conversion.
- Sobel gradients (Gx, Gy), gradient magnitude, gradient orientation.
- Patch-level feature extraction and PCA projection.
- Instability scoring from orientation entropy, Laplacian variance, and PCA spread.

What it shows:
- Whether local gradient structure looks physically coherent (camera-like) or unstable/high-frequency (common in synthetic denoising pipelines).
- Artifacts:
  - gradient magnitude map,
  - orientation visualization,
  - PCA projection plot.

### 2) Frequency Spectrum and Ringing Analysis (Classical)
What is used:
- 2D Fourier transform (DFT) over luminance image.
- Log-spectrum magnitude artifact (centered DC).
- Radial energy profile from low to high frequency bands.
- Interpretable metrics:
  - high-band energy ratio,
  - ringing proxy from profile oscillation,
  - directional anisotropy in the Fourier domain.

What it shows:
- Whether the image has unusual frequency-energy balance or oscillatory artifacts that can emerge from synthetic denoising/generation pipelines.
- Artifacts:
  - log-spectrum heatmap,
  - radial frequency profile plot.

### 3) Vanishing Point Geometry Consistency (Classical)
What is used:
- Canny edges + probabilistic Hough line segments.
- Pairwise line intersections (excluding near-parallel pairs).
- Grid-based clustering of intersection density.
- Consistency scoring from cluster concentration, intersection dispersion, and cluster entropy.

What it shows:
- Whether scene perspective converges to compact, coherent vanishing-point structure (more camera-like) or spreads across many inconsistent vanishing points (often synthetic-composite behavior).
- Artifacts:
  - line overlay image,
  - vanishing-point density map.

### 4) Sensor Noise and PRNU/CFA Consistency Checks (Classical)
What is used:
- High-frequency residual extraction via denoising and subtraction.
- Residual stationarity analysis over local tiles.
- Cross-channel residual correlation (B/G/R residual consistency).
- 2x2 CFA phase-energy analysis (periodicity proxy).

What it shows:
- Whether residual noise behaves like camera-pipeline traces (more structured/stationary with plausible CFA signatures) versus synthetic residual behavior.
- Artifacts:
  - residual noise map,
  - CFA 2x2 periodicity energy plot.

### 5) JPEG and Compression Forensics (Classical)
What is used:
- 8x8 block-boundary discontinuity analysis.
- Block-boundary versus interior gradient ratio.
- 8x8 DCT transform statistics on AC coefficients.
- DCT histogram periodicity and high-frequency near-zero ratio.

What it shows:
- Whether compression and quantization behavior looks unusually block-structured or histogram-periodic in ways often associated with synthetic generation pipelines.
- Artifacts:
  - block artifact heatmap,
  - DCT histogram plot.

### 6) Optical/Photometric Plausibility Checks (Classical)
What is used:
- Illumination coherence analysis from gradient orientation entropy.
- Low-frequency illumination map estimation.
- Cross-channel edge-alignment analysis using phase correlation.
- Tile-level edge-shift consistency metrics.

What it shows:
- Whether lighting flow and lens/channel alignment look physically plausible for a single camera capture.
- Artifacts:
  - estimated illumination map,
  - edge-channel shift diagnostic image.

### 7) Latent-Manifold Module (Optional Non-Classical Add-On)
What is used:
- Patch-level latent-manifold projection using PCA as a lightweight non-classical reconstruction proxy.
- Image reconstruction from latent coordinates and reconstruction-error heatmap extraction.

What it shows:
- Estimates how strongly the image adheres to a low-dimensional latent manifold by measuring reconstruction error.
- This module is explicitly marked non-classical and is optional.

## Next Iterations
- Improve score calibration using dataset-based threshold fitting.
- Replace latent stub with optional ONNX-based module.
