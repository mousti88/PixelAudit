#!/usr/bin/env python3
"""Run PixelAudit CLI over local folders and compute baseline metrics."""

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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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


def run_cli(cli_path: Path, image: Path) -> float:
    with tempfile.TemporaryDirectory(prefix="pixelaudit_eval_") as td:
        out_dir = Path(td)
        cmd = [str(cli_path), "--input", str(image), "--output-dir", str(out_dir)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        report = json.loads((out_dir / "report.json").read_text())
        return float(report["ai_probability_percent"]) / 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("build/pixelaudit_cli"))
    parser.add_argument("--limit-per-class", type=int, default=100)
    args = parser.parse_args()

    ai_images = collect_images(args.data_root / "ai")[: args.limit_per_class]
    real_images = collect_images(args.data_root / "real")[: args.limit_per_class]

    labels: list[int] = []
    scores: list[float] = []

    for image in ai_images:
        labels.append(1)
        scores.append(run_cli(args.cli, image))

    for image in real_images:
        labels.append(0)
        scores.append(run_cli(args.cli, image))

    auc = roc_auc(labels, scores)
    precision, recall, f1, acc = precision_recall_f1(labels, scores)
    calib = ece(labels, scores)

    print("Evaluation summary")
    print(f"Samples: {len(labels)}")
    print(f"ROC-AUC: {auc:.4f}" if not math.isnan(auc) else "ROC-AUC: NaN")
    print(f"Precision@0.5: {precision:.4f}")
    print(f"Recall@0.5: {recall:.4f}")
    print(f"F1@0.5: {f1:.4f}")
    print(f"Accuracy@0.5: {acc:.4f}")
    print(f"ECE(10 bins): {calib:.4f}" if not math.isnan(calib) else "ECE: NaN")


if __name__ == "__main__":
    main()
