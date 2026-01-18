"""
helper to read best_checkpoint.json and emit the model path for scripts or env export

usage:
    python -m ml.inference.load_best_checkpoint --output-dir ml/saved_weights --print-env

helps set ASL_MODEL_ID for the inference service and django pipeline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="print the best checkpoint path from best_checkpoint.json")
    parser.add_argument(
        "--output-dir",
        default="ml/saved_weights",
        help="training output directory containing best_checkpoint.json (default: ml/saved_weights)",
    )
    parser.add_argument("--print-env", action="store_true", help="emit export ASL_MODEL_ID=... for shell usage")
    return parser.parse_args()


def resolve_best_checkpoint_model_id(
    output_dir: str | Path = "ml/saved_weights",
    *,
    project_root: str | Path | None = None,
) -> Optional[str]:
    """
    Return the best checkpoint model id/path from best_checkpoint.json if available.

    - If best_checkpoint.json is missing, returns None.
    - If the checkpoint value looks like a local path and exists, returns an absolute path.
    - Otherwise returns the raw string (could be an HF repo id).
    """
    root = Path(project_root) if project_root is not None else None
    out_dir_path = Path(output_dir)
    if root is not None and not out_dir_path.is_absolute():
        out_dir_path = root / out_dir_path

    config_path = out_dir_path / "best_checkpoint.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    checkpoint = payload.get("best_checkpoint")
    if not checkpoint or not isinstance(checkpoint, str):
        return None

    cp = Path(checkpoint)
    if cp.is_absolute() and cp.exists():
        return str(cp)

    if root is not None:
        candidate = root / cp
        if candidate.exists():
            return str(candidate.resolve())

    if cp.exists():
        return str(cp.resolve())

    return checkpoint


def main() -> None:
    args = parse_args()
    checkpoint = resolve_best_checkpoint_model_id(args.output_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"best_checkpoint.json not found in {args.output_dir}")

    if args.print_env:
        print(f"export ASL_MODEL_ID={checkpoint}")
    else:
        print(checkpoint)


if __name__ == "__main__":
    main()
