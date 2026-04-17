import json
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="PixelAudit", layout="wide")

st.markdown(
    """
    <style>
    .stButton > button {
        background: linear-gradient(135deg, #18e6a1 0%, #2a77ff 100%);
        color: #ffffff;
        border: 1px solid #2a77ff;
        border-radius: 10px;
        font-weight: 700;
        box-shadow: 0 8px 22px rgba(30, 94, 255, 0.28);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #184fe0 0%, #2369e8 100%);
        color: #ffffff;
        border: 1px solid #1345c4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PixelAudit - Transparent AI Image Forensics")
st.caption("Classical CV-first detector with per-test evidence and artifacts")

left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    enable_latent = st.checkbox(
        "Enable optional non-classical latent-manifold add-on",
        value=False,
    )

    run_clicked = st.button("Run PixelAudit Tests", use_container_width=True)

    if uploaded is not None:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

with right_col:
    st.subheader("Results")
    result_placeholder = st.empty()

if run_clicked:
    if uploaded is None:
        st.error("Please upload an image first.")
    else:
        with tempfile.TemporaryDirectory(prefix="pixelaudit_ui_") as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / uploaded.name
            input_path.write_bytes(uploaded.getbuffer())

            output_dir = tmp_dir / "run_output"
            output_dir.mkdir(parents=True, exist_ok=True)

            default_cli = Path(__file__).resolve().parents[1] / "build" / "pixelaudit_cli"
            cli_path = Path(os.environ.get("PIXELAUDIT_CLI", str(default_cli)))

            cmd = [
                str(cli_path),
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
            ]

            calibration_file = os.environ.get("PIXELAUDIT_CALIBRATION_FILE", "").strip()
            if calibration_file:
                cmd.extend(["--calibration-file", calibration_file])

            if enable_latent:
                cmd.append("--enable-latent")

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except FileNotFoundError:
                st.error(
                    "CLI binary not found. Build C++ first and set PIXELAUDIT_CLI env var if needed."
                )
                st.stop()
            except subprocess.CalledProcessError as exc:
                st.error("PixelAudit CLI failed.")
                st.code(exc.stdout + "\n" + exc.stderr)
                st.stop()

            report_path = output_dir / "report.json"
            if not report_path.exists():
                st.error("Expected report.json was not produced.")
                st.code(proc.stdout + "\n" + proc.stderr)
                st.stop()

            report = json.loads(report_path.read_text())

            with result_placeholder.container():
                st.metric("AI Probability", f"{report['ai_probability_percent']:.2f}%")
                st.metric("Confidence", f"{report['confidence'] * 100:.1f}%")
                st.caption(f"Elapsed: {report['elapsed_ms']:.1f} ms")

                for idx, test in enumerate(report["tests"], start=1):
                    title = f"Test {idx}: {test['name']} | Score: {test['score_percent']:.1f}% | Status: {test['status']}"
                    with st.expander(title, expanded=False):
                        if test.get("is_non_classical", False):
                            st.warning("This test is explicitly non-classical.")

                        st.write(test.get("description", ""))
                        st.info(test.get("evidence_summary", "No explanation available."))

                        metrics = test.get("raw_metrics", {})
                        if metrics:
                            st.write("Raw metrics")
                            st.json(metrics)

                        artifact_paths = test.get("artifact_paths", [])
                        if artifact_paths:
                            st.write("Artifacts")
                            for artifact in artifact_paths:
                                p = Path(artifact)
                                if p.exists():
                                    st.image(str(p), caption=p.name, use_container_width=True)
                                else:
                                    st.caption(f"Missing artifact: {artifact}")

                st.write("Contribution breakdown")
                st.json(report.get("contribution_percent", {}))

st.markdown("---")
st.caption("Tip: set PIXELAUDIT_CLI=/absolute/path/to/pixelaudit_cli if binary is in a custom location.")
st.caption(
    "Optional: set PIXELAUDIT_CALIBRATION_FILE=/absolute/path/to/pixelaudit_calibration.json to use calibrated fusion."
)
