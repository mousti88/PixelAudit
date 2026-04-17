# Roadmap: Transparent AI Image Detection (Classical CV)

## 1) Project Vision
Build a transparent AI-vs-real image detector where:
- Core detection engine is C++ (C++20) with OpenCV.
- Every test is interpretable and produces:
  - test-specific score (0-100),
  - pass/fail state,
  - plain-language explanation,
  - optional visual artifact (proof image).
- Final output is a calibrated probability percentage that image is AI-generated, plus test breakdown.

Non-goal for v1:
- No deep classifier training as primary detector.
- No black-box single-number output without explanation.

## 2) Naming Suggestions
Top picks:
1. LumiTrace
2. VeriPixel
3. PhotoForensics++
4. TrueShot Check
5. LensProof
6. RealSight CV
7. PixelAudit
8. SceneIntegrity
9. FrameEvidence
10. AuthentiVision

Recommendation:
- Pick LumiTrace for a modern, research-friendly identity.
- Repository name example: lumitrace-ai-image-forensics

## 3) High-Level Architecture
## 3.1 Components
- C++ Core Library (OpenCV):
  - Loads image, validates type/shape.
  - Runs test pipeline sequentially.
  - Produces TestReport objects and FinalReport JSON.
  - Saves per-test visualization images.
- C++ CLI App:
  - Input image path.
  - Executes all enabled tests.
  - Outputs JSON + generated artifact image paths.
- UI Layer (Python, lightweight):
  - Two-column modern UI.
  - Left: upload image + run button.
  - Right: collapsible test sections in execution order.
  - Each section shows score, explanation, and artifact image when available.
- Evaluation Tools:
  - Dataset loader helper for Parveshiiii/AI-vs-Real.
  - Batch runner to measure AUC, precision/recall, calibration.

## 3.2 Interface Contract
Each test implements a shared interface:
- name
- description
- run(input_image) -> TestResult
- TestResult fields:
  - score_percent (0-100)
  - confidence (0-1)
  - status (pass/fail/inconclusive)
  - evidence_summary (human-readable)
  - artifact_paths (0..n)
  - raw_metrics (key-value map)

## 4) Research-Driven Classical Methods (Initial Test Suite)
Tests currently in scope (your list + additions):

1. Luminance-Gradient PCA Analysis
- Signal: gradient coherence differences between camera-captured and diffusion images.
- Method:
  - Convert RGB -> luminance.
  - Compute Sobel gradients (Gx, Gy), magnitude, orientation.
  - Form feature matrix from gradient statistics/patch distributions.
  - PCA projection and score based on distance to real-reference distribution.
- Artifact outputs:
  - gradient magnitude map,
  - orientation field visualization,
  - PCA projection plot image.

2. Vanishing Point Geometry Consistency
- Signal: real perspective tends to obey fewer coherent dominant vanishing points.
- Method:
  - Detect line segments (LSD / Hough).
  - Estimate intersections and cluster vanishing points.
  - Score inconsistency from spread/entropy of VP clusters.
- Artifact outputs:
  - line overlay image,
  - vanishing point cluster visualization.

3. Latent-Manifold Reconstruction Error (Classical-Compatible Variant)
- Note: this method uses a pretrained autoencoder and is not purely classical CV.
- v1 decision:
  - Keep as optional baseline module, disabled by default in strict-classical mode.
  - If enabled, clearly label as representation-learning-based.
- Artifact outputs:
  - reconstruction image,
  - error heatmap.

4. Frequency Spectrum and Ringing Analysis
- Signal: synthetic images often show unusual frequency energy distributions.
- Method:
  - FFT on luminance.
  - Radial energy profile and anisotropy checks.
  - Detect periodic denoising signatures and over-smooth/high-frequency imbalances.
- Artifact outputs:
  - log-spectrum image,
  - radial profile plot.

5. Sensor Noise and PRNU/CFA Consistency Checks
- Signal: real camera pipeline leaves structured sensor traces; generated images usually do not.
- Method:
  - Denoise and compute residual.
  - Analyze noise stationarity, color channel residual correlations.
  - Optional demosaicing/CFA periodicity tests.
- Artifact outputs:
  - noise residual map,
  - CFA periodicity map/plot.

6. JPEG and Compression Forensics
- Signal: inconsistencies in quantization/block statistics can reveal synthetic or edited generation pipelines.
- Method:
  - Block boundary and DCT-statistic features.
  - Double-compression signature checks.
- Artifact outputs:
  - block artifact heatmap,
  - DCT histogram plot.

7. Optical/Photometric Plausibility Checks
- Signal: generated scenes may violate lens and illumination consistency.
- Method:
  - Local illumination direction coherence.
  - Chromatic aberration and edge-channel shift consistency.
- Artifact outputs:
  - estimated illumination map,
  - edge shift diagnostic image.

## 5) Scoring and Transparency Design
## 5.1 Per-Test Score
Each test outputs score_percent with explicit formula in docs.
Example policy:
- 0-30: evidence favors real.
- 31-69: uncertain/mixed.
- 70-100: evidence favors AI generation.

## 5.2 Final Probability
Use weighted fusion of normalized test scores:
- final_logit = sum(weight_i * calibrated_score_i) + bias
- final_probability = sigmoid(final_logit)

Transparency requirement:
- Show each test contribution to final result.
- Display contribution bar chart and textual explanation:
  - strongest evidence for AI,
  - strongest evidence for real,
  - inconclusive tests.

## 5.3 Strict Explainability Rules
- No hidden features in final decision.
- Every scalar used in final score must be present in report JSON.
- Every test must include understandable rationale text.

## 6) UI/UX Plan (Two-Column)
Left column:
- Image uploader
- Run tests button
- Global summary card:
  - AI probability (%),
  - confidence,
  - elapsed time.

Right column:
- Collapsible sections: Test 1, Test 2, ...
- Within each section:
  - score + status badge,
  - short interpretation,
  - raw metrics table,
  - artifact image gallery when available.

Suggested stack:
- Streamlit for rapid transparent dashboard in early iterations.
- C++ CLI invoked from Python subprocess, exchanging JSON and artifacts.

## 7) Dataset and Evaluation Plan
Initial dataset:
- Hugging Face: Parveshiiii/AI-vs-Real

Workflow:
1. Build a small data prep helper to export samples into local folders.
2. Run batch evaluation using C++ detector.
3. Compute metrics:
  - ROC-AUC,
  - PR-AUC,
  - F1 at operating threshold,
  - calibration error (ECE).
4. Produce per-test ablation report:
  - impact of each test on final performance,
  - interpretability quality notes.

## 8) CMake and Build Reliability Plan (Low Friction)
Goal: avoid setup pain and make builds reproducible.

Plan:
1. Use CMakePresets.json for one-command configure/build/test.
2. Prefer package manager integration:
  - macOS: Homebrew OpenCV for quick start,
  - optional vcpkg manifest mode for reproducible CI.
3. Enforce target-based CMake:
  - detector_core library,
  - detector_cli executable,
  - optional tests target.
4. Add sanity checks at configure time:
  - OpenCV found/version,
  - compiler C++20 support,
  - OpenMP availability.
5. Add CI workflow to verify build on macOS and Linux.

## 9) Iteration Plan (What We Build Next)
Iteration 1: Foundation
- Repository skeleton, CMake presets, detector interfaces, JSON reporting.
- Implement Test 1 (Luminance-Gradient PCA) fully with artifact outputs.
- Minimal UI prototype with upload + collapsible test panel.

Iteration 2: Geometry and Frequency
- Implement vanishing point test and frequency-spectrum test.
- Add weighted fusion and contribution explanations.

Iteration 3: Forensics Expansion
- Add noise/CFA and JPEG forensics tests.
- Improve calibration and uncertainty handling.

Iteration 4: Evaluation and Hardening
- Batch evaluation pipeline on dataset.
- Performance optimization and memory profiling.
- Documentation and reproducibility hardening.

## 10) Acceptance Criteria for v1
- User uploads an image and receives final AI probability percentage.
- At least 3 classical tests run sequentially and appear as collapsible sections.
- Each section shows score, explanation, and artifact image (where applicable).
- Final report includes per-test contributions and raw metrics JSON.
- Build succeeds with documented commands on macOS.

## 11) OpenCV Implementation Recommendations
1. Pre-allocate Mats with create() and reuse buffers in loops.
2. In hot loops, use ptr<T>(row) instead of at<T>(y, x).
3. Keep image conversions explicit (CV_8UC3 <-> CV_32FC1) and documented.
4. Separate visualization code from scoring logic for testability.
5. Guard every external input with empty(), type, and channel checks.
6. Use cv::UMat only after benchmarking; do not assume speedup.
7. Keep deterministic pipelines (fixed kernels, fixed thresholds unless calibrated).
8. Build small synthetic unit tests for geometry/frequency edge cases.

## 12) Risks and Mitigations
- Risk: Method fails on unseen generators.
  - Mitigation: modular test fusion + easy addition of new forensic tests.
- Risk: False positives on heavy post-processing.
  - Mitigation: uncertainty band and per-test disagreement reporting.
- Risk: setup complexity on local machines.
  - Mitigation: presets + bootstrap script + CI-validated instructions.

## 13) Implementation Decisions (Confirmed)
1. Project name: PixelAudit
2. Detection scope: strict-classical core with optional latent-manifold module as explicit non-classical add-on
3. v1 UI stack: Streamlit dashboard

Implementation status:
- Iteration 1 has started with a working C++ core pipeline, first classical test, report generation, and Streamlit integration.
