#!/usr/bin/env python3
"""Calibrate PixelAudit fusion weights and bias from a labeled dataset."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path


def collect_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def roc_auc(labels: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    pos = 0
    neg = 0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += rank
            pos += 1
        else:
            neg += 1
    if pos == 0 or neg == 0:
        return float("nan")
    return (rank_sum - (pos * (pos + 1) / 2.0)) / (pos * neg)


def precision_recall_f1(labels: list[int], scores: list[float], threshold: float = 0.5):
    tp = fp = tn = fn = 0
    for y, s in zip(labels, scores):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return precision, recall, f1, accuracy


def ece(labels: list[int], scores: list[float], bins: int = 10) -> float:
    total = len(labels)
    if total == 0:
        return float("nan")
    ece_val = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        idxs = [j for j, s in enumerate(scores) if lo <= s < hi or (i == bins - 1 and s == 1.0)]
        if not idxs:
            continue
        acc = sum(labels[j] for j in idxs) / len(idxs)
        conf = sum(scores[j] for j in idxs) / len(idxs)
        ece_val += (len(idxs) / total) * abs(acc - conf)
    return ece_val


def best_f1_threshold(labels: list[int], scores: list[float]) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for i in range(1, 100):
        t = i / 100.0
        _, _, f1, _ = precision_recall_f1(labels, scores, threshold=t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def run_cli_report(cli_path: Path, image: Path, calibration_file: Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="pixelaudit_calib_") as td:
        out_dir = Path(td)
        cmd = [str(cli_path), "--input", str(image), "--output-dir", str(out_dir)]
        if calibration_file is not None:
            cmd.extend(["--calibration-file", str(calibration_file)])
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return json.loads((out_dir / "report.json").read_text())


def to_centered(score_percent: float) -> float:
    return (score_percent / 100.0 - 0.5) * 4.0


def predict_probs(features: list[list[float]], weights: list[float], bias: float) -> list[float]:
    s = sum(weights) + 1e-9
    probs: list[float] = []
    for row in features:
        num = sum(w * x for w, x in zip(weights, row))
        probs.append(sigmoid(bias + num / s))
    return probs


def fit_nonnegative_logistic(
    features: list[list[float]], labels: list[int], epochs: int, lr: float, l2: float
) -> tuple[list[float], float]:
    n = len(features)
    m = len(features[0]) if n else 0

    weights = [1.0 for _ in range(m)]
    bias = 0.0

    for _ in range(epochs):
        s = sum(weights) + 1e-9
        nums = [sum(w * x for w, x in zip(weights, row)) for row in features]
        probs = [sigmoid(bias + num / s) for num in nums]

        db = sum(p - y for p, y in zip(probs, labels)) / max(1, n)
        grad_w = [0.0 for _ in range(m)]

        for j in range(m):
            g = 0.0
            for i in range(n):
                dz_dw = (features[i][j] * s - nums[i]) / (s * s)
                g += (probs[i] - labels[i]) * dz_dw
            g /= max(1, n)
            g += 2.0 * l2 * (weights[j] - 1.0)
            grad_w[j] = g

        bias -= lr * db
        for j in range(m):
            weights[j] = max(0.0, weights[j] - lr * grad_w[j])

    return weights, bias


def dataset_exists(data_root: Path) -> bool:
    return (data_root / "ai").exists() and (data_root / "real").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/ai_vs_real"))
    parser.add_argument("--cli", type=Path, default=Path("build/pixelaudit_cli"))
    parser.add_argument("--limit-per-class", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/calibration/pixelaudit_calibration.json"),
    )
    args = parser.parse_args()

    if not dataset_exists(args.data_root):
        raise SystemExit(
            f"Dataset folders not found under {args.data_root}. Expected ai/ and real/."
        )

    ai_images = collect_images(args.data_root / "ai")[: args.limit_per_class]
    real_images = collect_images(args.data_root / "real")[: args.limit_per_class]

    if not ai_images or not real_images:
        raise SystemExit("Need at least one image in both ai/ and real/ folders.")

    labels: list[int] = []
    baseline_probs: list[float] = []
    features: list[list[float]] = []
    test_ids: list[str] = []

    all_images = [(1, p) for p in ai_images] + [(0, p) for p in real_images]
    for idx, (label, img) in enumerate(all_images, start=1):
        report = run_cli_report(args.cli, img)

        if not test_ids:
            test_ids = [t["test_id"] for t in report["tests"]]

        row: list[float] = []
        for t in report["tests"]:
            row.append(to_centered(float(t["score_percent"])))

        labels.append(label)
        baseline_probs.append(float(report["ai_probability_percent"]) / 100.0)
        features.append(row)

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(all_images)} images...")

    weights, bias = fit_nonnegative_logistic(
        features, labels, epochs=args.epochs, lr=args.lr, l2=args.l2
    )

    calibrated_probs = predict_probs(features, weights, bias)
    threshold, best_f1 = best_f1_threshold(labels, calibrated_probs)

    base_auc = roc_auc(labels, baseline_probs)
    base_prec, base_rec, base_f1, base_acc = precision_recall_f1(labels, baseline_probs)
    base_ece = ece(labels, baseline_probs)

    cal_auc = roc_auc(labels, calibrated_probs)
    cal_prec, cal_rec, cal_f1, cal_acc = precision_recall_f1(
        labels, calibrated_probs, threshold=threshold
    )
    cal_ece = ece(labels, calibrated_probs)

    out = {
        "fusion_bias": bias,
        "weights": {k: v for k, v in zip(test_ids, weights)},
        "recommended_threshold": threshold,
        "dataset": str(args.data_root),
        "samples": len(labels),
        "metrics": {
            "baseline": {
                "auc": base_auc,
                "precision@0.5": base_prec,
                "recall@0.5": base_rec,
                "f1@0.5": base_f1,
                "accuracy@0.5": base_acc,
                "ece": base_ece,
            },
            "calibrated": {
                "auc": cal_auc,
                "precision@threshold": cal_prec,
                "recall@threshold": cal_rec,
                "f1@threshold": cal_f1,
                "accuracy@threshold": cal_acc,
                "ece": cal_ece,
                "threshold": threshold,
                "best_f1": best_f1,
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print("Calibration complete")
    print(f"Output: {args.output}")
    print(f"Samples: {len(labels)}")
    print(f"Baseline AUC: {base_auc:.4f}")
    print(f"Calibrated AUC: {cal_auc:.4f}")
    print(f"Baseline F1@0.5: {base_f1:.4f}")
    print(f"Calibrated F1@{threshold:.2f}: {cal_f1:.4f}")
    print(f"Baseline ECE: {base_ece:.4f}")
    print(f"Calibrated ECE: {cal_ece:.4f}")


if __name__ == "__main__":
    main()
