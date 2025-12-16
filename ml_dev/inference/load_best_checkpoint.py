"""
helper to read best_checkpoint.json and emit the model path for scripts or env export

usage:
    python -m ml_dev.inference.load_best_checkpoint --output-dir ml_dev/saved_weights --print-env

helps set ASL_MODEL_ID for the inference service and django pipeline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def get_best_checkpoint_path(output_dir: str | Path = "ml_dev/saved_weights") -> str:
    """
    Read best_checkpoint.json and return the absolute path to the best model checkpoint.

    Args:
        output_dir: Directory containing best_checkpoint.json (can be relative or absolute)

    Returns:
        Absolute path to the best checkpoint directory
    """
    output_dir = Path(output_dir).resolve()
    config_path = output_dir / "best_checkpoint.json"
    if not config_path.exists():
        raise FileNotFoundError(f"best_checkpoint.json not found in {output_dir}")

    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    checkpoint = payload.get("best_checkpoint")
    if not checkpoint:
        raise ValueError(f"best_checkpoint not present in {config_path}")

    # If checkpoint is relative, resolve it from project root
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        # Assume checkpoint paths in JSON are relative to project root
        project_root = output_dir.parents[
            1
        ]  # Go up from ml_dev/saved_weights to project root
        checkpoint_path = (project_root / checkpoint).resolve()

    return str(checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="print the best checkpoint path from best_checkpoint.json"
    )
    parser.add_argument(
        "--output-dir",
        default="ml_dev/saved_weights",
        help="training output directory containing best_checkpoint.json (default: ml_dev/saved_weights)",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="emit export ASL_MODEL_ID=... for shell usage",
    )
    return parser.parse_args()


def main() -> None:
    print("loading best checkpoint...")
    args = parse_args()
    output_dir = Path(args.output_dir)
    config_path = output_dir / "best_checkpoint.json"
    if not config_path.exists():
        raise FileNotFoundError(f"best_checkpoint.json not found in {output_dir}")

    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    checkpoint = payload.get("best_checkpoint")
    if not checkpoint:
        raise ValueError(f"best_checkpoint not present in {config_path}")

    if args.print_env:
        print(f"export ASL_MODEL_ID={checkpoint}")
    else:
        print(checkpoint)


if __name__ == "__main__":
    main()
