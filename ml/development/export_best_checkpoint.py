"""
select the best fine tune checkpoint based on saved metrics files

usage:
    python -m ml.development.export_best_checkpoint --output-dir ml/saved_weights --metric accuracy

writes best_checkpoint.json that load_best_checkpoint.py and the service can consume
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def load_metrics(metrics_path: Path) -> Dict:
    with metrics_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def select_best_checkpoint(output_dir: Path, metric: str) -> Tuple[Path, Dict]:
    """
    select the best checkpoint using metrics_epoch_*.json files
    prefers higher metric values; if tie and loss exists, picks lower loss
    """
    best_path = None
    best_metrics = None

    for metrics_file in output_dir.glob("metrics_epoch_*.json"):
        metrics = load_metrics(metrics_file)
        value = metrics.get(metric)
        if value is None:
            continue

        if best_metrics is None:
            best_metrics = metrics
            best_path = metrics_file
            continue

        current_better = value > best_metrics.get(metric, float("-inf"))
        tie = value == best_metrics.get(metric)
        loss_better = metrics.get("loss", float("inf")) < best_metrics.get("loss", float("inf"))

        if current_better or (tie and loss_better):
            best_metrics = metrics
            best_path = metrics_file

    if best_path is None:
        raise FileNotFoundError(f"no metrics files found in {output_dir}")

    checkpoint_dir = output_dir / best_path.stem.replace("metrics_", "")
    return checkpoint_dir, best_metrics


def write_best_config(output_dir: Path, checkpoint_dir: Path, metrics: Dict) -> Path:
    """
    write a small json describing the chosen checkpoint
    """
    payload = {
        "best_checkpoint": str(checkpoint_dir),
        "metrics": metrics,
    }
    target = output_dir / "best_checkpoint.json"
    with target.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="select best checkpoint and emit config file")
    parser.add_argument("--output-dir", required=True, help="training output directory with metrics_epoch_*.json files")
    parser.add_argument("--metric", default="accuracy", help="metric name to maximize when selecting best checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir, metrics = select_best_checkpoint(output_dir, args.metric)
    target = write_best_config(output_dir, checkpoint_dir, metrics)
    print(f"best checkpoint: {checkpoint_dir}")
    print(f"metrics: {metrics}")
    print(f"wrote config: {target}")


if __name__ == "__main__":
    main()
