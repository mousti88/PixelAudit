#!/usr/bin/env python3
"""Evaluate PixelAudit per-audit performance on local AI/real folders."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from collections import defaultdict
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


def run_cli(cli_path: Path, image: Path, enable_latent: bool) -> dict:
	with tempfile.TemporaryDirectory(prefix="pixelaudit_eval_audits_") as td:
		out_dir = Path(td)
		cmd = [str(cli_path), "--input", str(image), "--output-dir", str(out_dir)]
		if enable_latent:
			cmd.append("--enable-latent")
		subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		return json.loads((out_dir / "report.json").read_text())


def to_probability(score_percent: float) -> float:
	return max(0.0, min(1.0, float(score_percent) / 100.0))


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-root", type=Path, required=True)
	parser.add_argument("--cli", type=Path, default=Path("build/pixelaudit_cli"))
	parser.add_argument("--limit-per-class", type=int, default=50)
	parser.add_argument("--enable-latent", action="store_true")
	parser.add_argument("--out-json", type=Path)
	args = parser.parse_args()

	ai_images = collect_images(args.data_root / "ai")[: args.limit_per_class]
	real_images = collect_images(args.data_root / "real")[: args.limit_per_class]

	if not ai_images or not real_images:
		raise SystemExit("Need non-empty ai/ and real/ image folders under --data-root")

	audit_scores: dict[str, list[float]] = defaultdict(list)
	audit_names: dict[str, str] = {}
	labels: list[int] = []

	for label, imgs in ((1, ai_images), (0, real_images)):
		for image in imgs:
			report = run_cli(args.cli, image, args.enable_latent)
			labels.append(label)
			tests = report.get("tests", [])
			for t in tests:
				tid = t["test_id"]
				audit_scores[tid].append(to_probability(t["score_percent"]))
				audit_names[tid] = t.get("name", tid)

	summary: dict[str, dict] = {}
	for tid, scores in audit_scores.items():
		if len(scores) != len(labels):
			continue
		auc = roc_auc(labels, scores)
		precision, recall, f1, acc = precision_recall_f1(labels, scores)
		ai_mean = sum(s for s, y in zip(scores, labels) if y == 1) / sum(labels)
		real_count = len(labels) - sum(labels)
		real_mean = sum(s for s, y in zip(scores, labels) if y == 0) / max(1, real_count)
		summary[tid] = {
			"name": audit_names.get(tid, tid),
			"samples": len(scores),
			"roc_auc": auc,
			"precision_at_0_5": precision,
			"recall_at_0_5": recall,
			"f1_at_0_5": f1,
			"accuracy_at_0_5": acc,
			"mean_ai_score": ai_mean,
			"mean_real_score": real_mean,
			"mean_gap_ai_minus_real": ai_mean - real_mean,
		}

	ranked = sorted(
		summary.items(),
		key=lambda kv: (-(kv[1]["roc_auc"] if not math.isnan(kv[1]["roc_auc"]) else -1.0)),
	)

	print("Per-audit evaluation")
	print(f"AI samples: {len(ai_images)} | Real samples: {len(real_images)}")
	for tid, metrics in ranked:
		auc = metrics["roc_auc"]
		auc_s = f"{auc:.4f}" if not math.isnan(auc) else "NaN"
		print(
			f"- {tid}: AUC={auc_s}, F1={metrics['f1_at_0_5']:.4f}, "
			f"gap={metrics['mean_gap_ai_minus_real']:.4f} "
			f"({metrics['mean_ai_score']:.3f} vs {metrics['mean_real_score']:.3f})"
		)

	output = {
		"ai_samples": len(ai_images),
		"real_samples": len(real_images),
		"enable_latent": bool(args.enable_latent),
		"audits": summary,
	}

	if args.out_json:
		args.out_json.parent.mkdir(parents=True, exist_ok=True)
		args.out_json.write_text(json.dumps(output, indent=2))
		print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
	main()
