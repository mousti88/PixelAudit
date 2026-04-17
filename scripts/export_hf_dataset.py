#!/usr/bin/env python3
"""Export a local subset of Parveshiiii/AI-vs-Real for quick evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def find_image_and_label_keys(sample: dict) -> tuple[str, str]:
    image_key = None
    label_key = None

    for key, value in sample.items():
        if image_key is None and hasattr(value, "save"):
            image_key = key
        if label_key is None and key.lower() in {
            "label",
            "labels",
            "class",
            "target",
            "binary_label",
        }:
            label_key = key

    if image_key is None:
        raise ValueError("Could not infer image key from dataset sample")
    if label_key is None:
        raise ValueError("Could not infer label key from dataset sample")

    return image_key, label_key


def is_ai_label(label_value, ai_label_value: int) -> bool:
    if isinstance(label_value, bool):
        return int(label_value) == ai_label_value
    if isinstance(label_value, int):
        return label_value == ai_label_value

    text = str(label_value).lower()
    if text.isdigit():
        return int(text) == ai_label_value
    return any(token in text for token in ["ai", "fake", "generated", "synthetic"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/ai_vs_real"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument(
        "--ai-label-value",
        type=int,
        default=0,
        help="Numeric label value representing AI-generated images (default: 0).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional per-class cap. If omitted, max-samples/2 is used.",
    )
    args = parser.parse_args()

    try:
        ds = load_dataset(
            "Parveshiiii/AI-vs-Real",
            split=args.split,
            verification_mode="no_checks",
        )
    except TypeError:
        ds = load_dataset(
            "Parveshiiii/AI-vs-Real",
            split=args.split,
            ignore_verifications=True,
        )
    if len(ds) == 0:
        raise RuntimeError("Dataset split is empty")

    sample = ds[0]
    image_key, label_key = find_image_and_label_keys(sample)

    ai_dir = args.output / "ai"
    real_dir = args.output / "real"
    ai_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    for p in ai_dir.glob("*.png"):
        p.unlink()
    for p in real_dir.glob("*.png"):
        p.unlink()

    per_class_limit = args.max_per_class
    if per_class_limit is None:
        per_class_limit = max(1, args.max_samples // 2)

    ai_count = 0
    real_count = 0
    count = 0
    for idx, item in enumerate(ds):
        image = item[image_key]
        label = item[label_key]

        if is_ai_label(label, args.ai_label_value):
            if ai_count >= per_class_limit:
                continue
            target_dir = ai_dir
            ai_count += 1
        else:
            if real_count >= per_class_limit:
                continue
            target_dir = real_dir
            real_count += 1

        image_path = target_dir / f"sample_{idx:06d}.png"
        image.save(image_path)
        count += 1

        if ai_count >= per_class_limit and real_count >= per_class_limit:
            break

    print(
        f"Exported {count} samples into {args.output} "
        f"(ai={ai_count}, real={real_count})"
    )


if __name__ == "__main__":
    main()
