"""
comprehensive evaluation utilities for asl models

this module loads the hugging face asl dataset and runs accuracy, confusion,
and confidence analyses on any predictor that implements baseaslmodel.
intended use: outside the training loop to sanity check new checkpoints;
defaults to aslpredictor if no model is supplied.

run (from repo root):
- set ASL_MODEL_ID to a checkpoint or pass --model-id
- python -m ml.testing.evaluator --model-id ml/saved_weights/epoch_7 --num-samples 50 --split train

export confusion matrix:
- python -m ml.testing.evaluator --model-id ml/saved_weights/epoch_7 --num-samples 200 --split test --export-dir ml/artifacts/eval --export-formats csv,json,png

evaluate the full split (all samples/classes present):
- python -m ml.testing.evaluator --split test --all-samples

evaluate a stratified subset (e.g., 100 samples per class):
- python -m ml.testing.evaluator --all-samples --per-class-samples 100

eval a specific sample size on GPU (Apple Silicon):
- python -m ml.testing.evaluator --device mps --per-class-samples 100

by default, artifacts are saved under ml/artifacts/eval (disable with --no-save)
"""

from __future__ import annotations

from pathlib import Path
import sys
import os
import argparse
import csv
import json
import random
import logging
import time
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

# add project root to sys.path for direct script runs
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "ml" / "artifacts" / "mplconfig"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from PIL import Image
from ml.development.clip_base import BaseASLModel
from ml.inference.clip_asl_inference import ASLPredictor
from ml.inference.load_best_checkpoint import resolve_best_checkpoint_model_id

logger = logging.getLogger(__name__)


def load_best_checkpoint_metadata(project_root: Path = PROJECT_ROOT) -> Optional[Dict[str, Any]]:
    """
    Load training metadata from `ml/saved_weights/best_checkpoint.json` (if present).

    This is used purely for reporting (matplotlib dashboard + stdout), so evaluator runs do
    not depend on it being present.

    Args:
        project_root: repository root, used to resolve the JSON path.

    Returns:
        Parsed JSON dict (with keys like `best_checkpoint` and `metrics`) or None if the
        file is missing/unreadable.
    """
    config_path = project_root / "ml" / "saved_weights" / "best_checkpoint.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.info("Failed to read best checkpoint metadata from %s: %s", config_path, exc)
        return None


def log_torch_device_diagnostics(requested_device: Optional[str]) -> None:
    """
    Log helpful torch device diagnostics to the terminal.

    This is intended to answer: "Did CUDA/MPS initialize, and what device will we run on?"
    It prints availability, basic device info, and performs a tiny allocation smoke-check.

    Args:
        requested_device: explicit device string like "cpu", "mps", "cuda", or None for auto.
    """
    requested = requested_device or "auto"
    logger.info("Requested device: %s", requested)
    logger.info("torch=%s", getattr(torch, "__version__", "unknown"))

    cuda_available = torch.cuda.is_available()
    logger.info("cuda available: %s", cuda_available)
    if cuda_available:
        try:
            logger.info("cuda device count: %s", torch.cuda.device_count())
            logger.info("cuda device name: %s", torch.cuda.get_device_name(0))
        except Exception as exc:
            logger.info("cuda details unavailable: %s", exc)

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(getattr(mps_backend, "is_built", lambda: False)()) if mps_backend else False
    mps_available = bool(getattr(mps_backend, "is_available", lambda: False)()) if mps_backend else False
    logger.info("mps built: %s", mps_built)
    logger.info("mps available: %s", mps_available)

    # Smoke-check the requested device (or inferred target) can allocate.
    target = requested_device
    if target is None:
        if cuda_available:
            target = "cuda"
        elif mps_available:
            target = "mps"
        else:
            target = "cpu"

    try:
        _ = torch.zeros(1, device=target)
        logger.info("device smoke-check OK: %s", target)
    except Exception as exc:
        logger.exception("device smoke-check FAILED for %s: %s", target, exc)


class ASLEvaluator:
    """
    Evaluate an ASL predictor on a dataset split.

    Design notes:
    - Runs a single inference pass over selected indices (optionally stratified).
    - Derives all metrics/plots from cached arrays to avoid repeated forward passes.
    - Intended for checkpoint sanity-checking and quick analysis (CLI + matplotlib).
    """
    
    def __init__(self, model: Optional[BaseASLModel] = None, dataset_name: str = "aliciiavs/sign_language_image_dataset"):
        """
        Args:
            model: Predictor implementing `predict()` (and optionally `predict_with_probs()`).
                Defaults to `ASLPredictor()` which loads from `ASL_MODEL_ID`/best checkpoint.
            dataset_name: HuggingFace dataset identifier.

        Attributes:
            log_every: Progress log interval (in samples).
            topk_metrics_k: K used for top-k accuracy when probabilities are available.
            ece_bins: Number of bins for the calibration curve / ECE computation.
        """
        self.model = model or ASLPredictor()
        self.dataset_name = dataset_name
        self.dataset = None
        self.split = None
        self.log_every = 100
        self.topk_metrics_k = 5
        self.ece_bins = 10

    def _resolve_split(self, requested_split: Optional[str]) -> str:
        """
        Resolve a user-requested split name to a split that exists in the loaded dataset.

        HuggingFace datasets are typically returned as a `DatasetDict` (e.g. train/validation/test),
        but some datasets ship only a single split. This helper:
        - Accepts common aliases (e.g. "test" -> "validation"/"val"/"dev" if needed).
        - Falls back to the only available split when the dataset has a single split.

        Args:
            requested_split: Split name requested by the caller/CLI.

        Returns:
            A split name that exists in `self.dataset`.
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded; call load_dataset() first.")

        available = list(self.dataset.keys())
        if not available:
            raise RuntimeError("Loaded dataset has no splits.")

        if requested_split is None:
            return "train" if "train" in available else available[0]

        if requested_split in self.dataset:
            return requested_split

        aliases: Dict[str, List[str]] = {
            "test": ["test", "validation", "valid", "val", "dev"],
            "val": ["validation", "valid", "val", "dev"],
            "valid": ["validation", "valid", "val", "dev"],
            "dev": ["validation", "valid", "val", "dev"],
            "train": ["train", "training"],
        }
        for candidate in aliases.get(requested_split, []):
            if candidate in self.dataset:
                return candidate

        if len(available) == 1:
            return available[0]

        raise ValueError(f"Unknown split '{requested_split}'. Available splits: {available}")

    def load_dataset(self, split: str = "train"):
        """
        Load the configured HuggingFace dataset and set the active split.

        Args:
            split: Desired split name (may be resolved via `_resolve_split`).
        """
        print(f"Loading {self.dataset_name} dataset...")
        self.dataset = load_dataset(self.dataset_name)
        resolved_split = self._resolve_split(split)
        if resolved_split != split:
            print(f"Requested split '{split}' not found; using '{resolved_split}' instead.")
        self.split = resolved_split
        print(f"Loaded {len(self.dataset[self.split])} samples from {self.split} split")

    def _eval_count(self, num_samples: Optional[int]) -> int:
        """
        Compute how many samples will be evaluated for this run.

        Args:
            num_samples: If None, evaluate the full split. Otherwise, evaluate up to
                `min(num_samples, len(split))`.

        Returns:
            Number of samples to evaluate.
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded; call load_dataset() first.")
        total = len(self.dataset[self.split])
        if num_samples is None:
            return total
        return min(num_samples, total)

    def select_indices(
        self,
        *,
        per_class_samples: Optional[int] = None,
        seed: int = 42,
        shuffle_within_class: bool = True,
    ) -> List[int]:
        """
        Return a list of dataset indices to evaluate.

        If per_class_samples is provided, selects up to that many samples per label
        (stratified by class). Otherwise returns all indices for the split.

        Notes:
        - This function only selects indices; it does not run inference.
        - With `seed`, the same selection is repeatable across runs as long as the dataset
          ordering is stable.
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded; call load_dataset() first.")

        ds = self.dataset[self.split]
        total = len(ds)

        if per_class_samples is None:
            return list(range(total))

        if per_class_samples <= 0:
            raise ValueError("--per-class-samples must be a positive integer")

        try:
            labels = ds["label"]
        except Exception:
            labels = [ds[i]["label"] for i in range(total)]

        by_label: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            by_label[int(label)].append(idx)

        rng = random.Random(seed)
        selected: List[int] = []
        for label in sorted(by_label.keys()):
            label_indices = by_label[label]
            if shuffle_within_class:
                rng.shuffle(label_indices)
            selected.extend(label_indices[:per_class_samples])

        rng.shuffle(selected)
        return selected

    def evaluate_comprehensive(
        self,
        num_samples: Optional[int] = 100,
        *,
        indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run a full evaluation and return a results dictionary.

        This method is the main entry point: it runs a single inference pass over the chosen
        indices and computes all derived metrics (confusion matrix, per-class metrics, calibration).

        Args:
            num_samples: If `indices` is not provided, evaluate the first N samples from the split.
                If None, evaluate the entire split.
            indices: Explicit dataset indices to evaluate. When provided, `num_samples` is ignored.

        Returns:
            A JSON-serializable dict with keys:
            - model_name, model_id
            - basic_evaluation (accuracy, avg confidence, runtime, etc.)
            - detailed_analysis (confusion, per-class stats, calibration, etc.)
            - num_samples (actual evaluated count)
        """
        if self.dataset is None:
            self.load_dataset()
        
        # If the dataset was injected externally, fall back to a sensible default split.
        if self.split is None:
            self.split = self._resolve_split(None)

        if indices is None:
            eval_n = self._eval_count(num_samples)
            indices = list(range(eval_n))
        else:
            indices = list(indices)
            eval_n = len(indices)

        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION - {self.model.__class__.__name__}")
        print(f"{'='*60}")

        inference = self._run_inference(indices)
        basic_results = self._evaluate_basic_from_inference(inference)
        detailed_results = self._analyze_detailed_from_inference(inference)
        
        results = {
            "model_name": self.model.__class__.__name__,
            "model_id": getattr(self.model, "model_name", None),
            "basic_evaluation": basic_results,
            "detailed_analysis": detailed_results,
            "num_samples": eval_n,
        }

        return results

    def _run_inference(self, indices: Sequence[int]) -> Dict[str, Any]:
        """
        Run model inference for the given dataset indices exactly once.

        This is intentionally the only place where we call the model in a loop. Everything else
        (confusion matrix, calibration, per-class metrics, difficult examples) is derived from the
        returned arrays.

        Args:
            indices: Dataset indices (relative to the active split) to evaluate.

        Returns:
            Dict of numpy arrays and metadata, including:
            - true_idx, pred_idx (int arrays of class indices)
            - confidences (float array of max softmax per sample)
            - correct_mask (bool array)
            - topk_correct_mask/topk_k when probability vectors are available
            - per_class_counts (true-label histogram)
            - elapsed_seconds, samples_per_second
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded; call load_dataset() first.")

        ds = self.dataset[self.split]
        eval_n = len(indices)
        letters: List[str] = list(getattr(self.model, "letters", []))
        if not letters:
            raise RuntimeError("Model must expose a `letters` list.")
        letter_to_index = {letter: i for i, letter in enumerate(letters)}

        k = max(1, int(self.topk_metrics_k))
        k = min(k, len(letters))

        true_idx: List[int] = []
        pred_idx: List[int] = []
        confidences: List[float] = []
        correct_mask: List[bool] = []
        topk_correct_mask: List[bool] = []
        sample_indices: List[int] = []
        per_class_counts: Dict[str, int] = defaultdict(int)
        top3_by_sample: Dict[int, Optional[List[Tuple[str, float]]]] = {}
        has_probs = False

        logger.info("Inference pass: %s samples", eval_n)
        t0 = time.perf_counter()

        for n, idx in enumerate(indices, start=1):
            sample = ds[idx]
            image = sample["image"]
            true_label = int(sample["label"])
            true_letter = letters[true_label]

            pred_letter, confidence, probs = self._predict(image)
            pred_label = letter_to_index.get(pred_letter)
            if pred_label is None:
                raise RuntimeError(f"Predicted letter '{pred_letter}' not found in model.letters")

            is_correct = pred_label == true_label
            is_topk_correct = False
            top3 = None
            if probs is not None:
                has_probs = True
                try:
                    top_probs, top_indices = torch.topk(probs[0], k=k)
                    top_index_list = [int(i) for i in top_indices.tolist()]
                    is_topk_correct = true_label in top_index_list

                    top3_k = min(3, len(top_index_list))
                    top3 = [
                        (letters[i], float(p))
                        for p, i in zip(top_probs[:top3_k].tolist(), top_index_list[:top3_k])
                    ]
                except Exception:
                    is_topk_correct = False
                    top3 = None
            top3_by_sample[idx] = top3

            true_idx.append(true_label)
            pred_idx.append(pred_label)
            confidences.append(float(confidence))
            correct_mask.append(bool(is_correct))
            topk_correct_mask.append(bool(is_topk_correct))
            sample_indices.append(idx)
            per_class_counts[true_letter] += 1

            if n <= 3:
                print(f"Sample {n} (idx {idx}): True={true_letter}, Pred={pred_letter}, Conf={confidence:.3f}")

            if self.log_every and (n % self.log_every == 0 or n == eval_n):
                elapsed = time.perf_counter() - t0
                rate = (n / elapsed) if elapsed > 0 else 0.0
                logger.info("Inference progress: %s/%s (%.1f%%), %.2f samples/s", n, eval_n, 100.0 * n / eval_n, rate)

        elapsed = time.perf_counter() - t0
        return {
            "letters": letters,
            "indices": sample_indices,
            "true_idx": np.array(true_idx, dtype=int),
            "pred_idx": np.array(pred_idx, dtype=int),
            "confidences": np.array(confidences, dtype=float),
            "correct_mask": np.array(correct_mask, dtype=bool),
            "topk_correct_mask": (np.array(topk_correct_mask, dtype=bool) if has_probs else None),
            "topk_k": (k if has_probs else None),
            "per_class_counts": dict(per_class_counts),
            "top3_by_sample": top3_by_sample,
            "elapsed_seconds": float(elapsed),
            "samples_per_second": float((eval_n / elapsed) if elapsed > 0 else 0.0),
        }

    def _evaluate_basic_from_inference(self, inference: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute basic summary metrics from the raw inference arrays.

        This includes accuracy, average confidence, top-k accuracy (when available),
        runtime, and simple per-class sample counts.

        Args:
            inference: Output dict from `_run_inference`.

        Returns:
            Dict stored under `results["basic_evaluation"]`.
        """
        correct_mask = inference["correct_mask"]
        confidences = inference["confidences"]
        indices = inference["indices"]
        letters = inference["letters"]
        true_idx = inference["true_idx"]
        pred_idx = inference["pred_idx"]
        topk_k = inference.get("topk_k")
        topk_correct_mask = inference.get("topk_correct_mask")

        eval_n = int(len(indices))
        correct = int(correct_mask.sum())
        accuracy = float(correct / eval_n) if eval_n else 0.0

        topk_accuracy = None
        if topk_correct_mask is not None:
            topk_accuracy = float(topk_correct_mask.mean()) if eval_n else 0.0

        predictions = []
        for i, sample_idx in enumerate(indices):
            t = letters[int(true_idx[i])]
            p = letters[int(pred_idx[i])]
            c = float(confidences[i])
            predictions.append(
                {
                    "true_letter": t,
                    "pred_letter": p,
                    "confidence": c,
                    "correct": bool(correct_mask[i]),
                    "sample_index": int(sample_idx),
                }
            )

        return {
            "accuracy": accuracy,
            "topk_k": topk_k,
            "topk_accuracy": topk_accuracy,
            "avg_confidence": float(np.mean(confidences)) if eval_n else 0.0,
            "correct_predictions": correct,
            "total_predictions": eval_n,
            "predictions": predictions,
            "per_class_counts": inference["per_class_counts"],
            "elapsed_seconds": inference["elapsed_seconds"],
            "samples_per_second": inference["samples_per_second"],
        }

    def _predict(self, image: Image.Image) -> Tuple[str, float, Optional[torch.Tensor]]:
        """
        Normalize predictor outputs to (letter, confidence, probs|None) across models.

        Supported predictor APIs:
        - `predict_with_probs(image) -> (letter, confidence, probs)` (preferred for top-k + calibration)
        - `predict(image) -> (letter, confidence)`
        - `predict(image) -> (letter, confidence, probs)`

        Returns:
            (predicted_letter, confidence, probs_or_none)
        """
        if hasattr(self.model, "predict_with_probs"):
            result = self.model.predict_with_probs(image)
        else:
            result = self.model.predict(image)
        if isinstance(result, tuple) and len(result) == 3:
            return result
        if isinstance(result, tuple) and len(result) == 2:
            letter, confidence = result
            return letter, confidence, None
        raise ValueError("Unexpected prediction format; expected tuple of length 2 or 3.")
    
    def _analyze_detailed_from_inference(self, inference: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derive detailed analysis artifacts from raw inference arrays.

        Computes:
        - Confusion matrix (counts + row-normalized)
        - Per-class accuracy/precision/recall/F1 + macro/micro aggregates
        - Calibration curve + ECE (using max-softmax confidence as the score)
        - Most-confused class pairs (with example indices)
        - Difficult samples list (low confidence and/or incorrect)

        Args:
            inference: Output dict from `_run_inference`.

        Returns:
            Dict stored under `results["detailed_analysis"]`.
        """
        letters: List[str] = inference["letters"]
        true_idx: np.ndarray = inference["true_idx"]
        pred_idx: np.ndarray = inference["pred_idx"]
        confidences: np.ndarray = inference["confidences"]
        correct_mask: np.ndarray = inference["correct_mask"]
        top3_by_sample: Dict[int, Optional[List[Tuple[str, float]]]] = inference["top3_by_sample"]

        num_classes = len(letters)
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        np.add.at(confusion, (true_idx, pred_idx), 1)

        row_sums = confusion.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            confusion_norm = np.divide(confusion, row_sums[:, None], out=np.zeros_like(confusion, dtype=float), where=row_sums[:, None] != 0)

        # precision/recall/f1 per class
        tp = np.diag(confusion).astype(float)
        fp = confusion.sum(axis=0).astype(float) - tp
        fn = confusion.sum(axis=1).astype(float) - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

        macro_precision = float(np.mean(precision)) if num_classes else 0.0
        macro_recall = float(np.mean(recall)) if num_classes else 0.0
        macro_f1 = float(np.mean(f1)) if num_classes else 0.0

        total = float(confusion.sum())
        micro_tp = float(tp.sum())
        micro_precision = float(micro_tp / total) if total else 0.0
        micro_recall = float(micro_tp / total) if total else 0.0
        micro_f1 = float(micro_tp / total) if total else 0.0

        # confidence analysis
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        confidence_analysis = {
            "all_confidences": confidences.tolist(),
            "correct_confidences": correct_conf.tolist(),
            "incorrect_confidences": incorrect_conf.tolist(),
            "mean_confidence": float(np.mean(confidences)) if len(confidences) else 0.0,
            "mean_correct_confidence": float(np.mean(correct_conf)) if len(correct_conf) else 0.0,
            "mean_incorrect_confidence": float(np.mean(incorrect_conf)) if len(incorrect_conf) else 0.0,
            "confidence_std": float(np.std(confidences)) if len(confidences) else 0.0,
        }

        # per-letter performance
        per_letter = {}
        for i, letter in enumerate(letters):
            total_i = int(confusion[i].sum())
            correct_i = int(confusion[i, i])
            per_letter[letter] = {
                "accuracy": float(correct_i / total_i) if total_i else 0.0,
                "total_samples": total_i,
                "correct_predictions": correct_i,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }

        # most confused pairs (off-diagonal)
        pairs = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    continue
                count = int(confusion[i, j])
                if count > 0:
                    pairs.append((count, letters[i], letters[j]))
        pairs.sort(reverse=True, key=lambda t: t[0])
        sample_indices = np.array(inference["indices"], dtype=int)
        most_confused_pairs = []
        for c, t, p in pairs[:10]:
            i = letters.index(t)
            j = letters.index(p)
            mask = (true_idx == i) & (pred_idx == j)
            examples = sample_indices[mask][:3].tolist()
            most_confused_pairs.append({"count": c, "true": t, "pred": p, "examples": examples})

        # calibration curve + ECE
        bins = max(2, int(self.ece_bins))
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        bin_ids = np.digitize(confidences, bin_edges, right=True) - 1
        bin_ids = np.clip(bin_ids, 0, bins - 1)
        bin_acc = np.zeros(bins, dtype=float)
        bin_conf = np.zeros(bins, dtype=float)
        bin_count = np.zeros(bins, dtype=int)
        for b in range(bins):
            mask = bin_ids == b
            cnt = int(mask.sum())
            bin_count[b] = cnt
            if cnt:
                bin_acc[b] = float(correct_mask[mask].mean())
                bin_conf[b] = float(confidences[mask].mean())
        ece = 0.0
        n_total = len(confidences)
        if n_total:
            for b in range(bins):
                if bin_count[b]:
                    ece += (bin_count[b] / n_total) * abs(bin_acc[b] - bin_conf[b])

        calibration = {
            "bins": bins,
            "bin_edges": bin_edges.tolist(),
            "bin_accuracy": bin_acc.tolist(),
            "bin_confidence": bin_conf.tolist(),
            "bin_count": bin_count.tolist(),
            "ece": float(ece),
        }

        # difficult samples: lowest-confidence incorrect or low-confidence
        difficult = []
        for i, sample_idx in enumerate(inference["indices"]):
            is_correct = bool(correct_mask[i])
            conf = float(confidences[i])
            if conf < 0.5 or not is_correct:
                true_letter = letters[int(true_idx[i])]
                pred_letter = letters[int(pred_idx[i])]
                top3 = top3_by_sample.get(int(sample_idx))
                difficult.append(
                    {
                        "index": int(sample_idx),
                        "true_letter": true_letter,
                        "pred_letter": pred_letter,
                        "confidence": conf,
                        "correct": is_correct,
                        "top3": top3,
                    }
                )
        difficult.sort(key=lambda x: x["confidence"])
        difficult_samples = difficult[:10]

        return {
            "confusion_matrix": confusion,
            "confusion_matrix_normalized": confusion_norm,
            "confidence_analysis": confidence_analysis,
            "calibration": calibration,
            "per_letter_performance": per_letter,
            "most_confused_pairs": most_confused_pairs,
            "difficult_samples": difficult_samples,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        }
    
    def visualize_results(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        *,
        show_seconds: float = 30.0,
    ):
        """
        Render a matplotlib dashboard for the current evaluation results.

        Layout:
        - Confusion matrix (counts)
        - Confusion matrix (row-normalized)
        - Confidence histogram (all/correct/incorrect)
        - Calibration curve + ECE
        - Per-letter accuracy bar chart
        - Model + runtime + summary metrics text panel

        Args:
            results: Dict returned by `evaluate_comprehensive`.
            save_path: If provided, the dashboard image is saved to this path.
            show_seconds: If >0, show non-blocking for this many seconds then auto-close.
                If 0, block until the window is closed (interactive usage).
        """
        print("Creating visualizations...")
        
        # set up plots
        fig, axes = plt.subplots(3, 2, figsize=(18, 16), constrained_layout=True)

        letters = list(getattr(self.model, "letters", []))
        confusion_matrix = results["detailed_analysis"]["confusion_matrix"]
        sns.heatmap(
            confusion_matrix,
            annot=False,
            cmap="Blues",
            xticklabels=letters,
            yticklabels=letters,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Confusion Matrix (Counts)")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("True")
        axes[0, 0].tick_params(axis="x", rotation=45)

        confusion_norm = results["detailed_analysis"].get("confusion_matrix_normalized")
        if confusion_norm is not None:
            sns.heatmap(
                confusion_norm,
                annot=False,
                vmin=0.0,
                vmax=1.0,
                cmap="Blues",
                xticklabels=letters,
                yticklabels=letters,
                ax=axes[0, 1],
            )
            axes[0, 1].set_title("Confusion Matrix (Row-Normalized)")
            axes[0, 1].set_xlabel("Predicted")
            axes[0, 1].set_ylabel("True")
            axes[0, 1].tick_params(axis="x", rotation=45)
        else:
            axes[0, 1].axis("off")

        confidence_analysis = results["detailed_analysis"]["confidence_analysis"]
        axes[1, 0].hist(confidence_analysis["all_confidences"], bins=20, alpha=0.7, label="All")
        axes[1, 0].hist(confidence_analysis["correct_confidences"], bins=20, alpha=0.7, label="Correct")
        axes[1, 0].hist(confidence_analysis["incorrect_confidences"], bins=20, alpha=0.7, label="Incorrect")
        axes[1, 0].set_title("Confidence Distribution")
        axes[1, 0].set_xlabel("Confidence Score")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()

        calib = results["detailed_analysis"].get("calibration") or {}
        if calib:
            bin_acc = np.array(calib.get("bin_accuracy", []), dtype=float)
            bin_conf = np.array(calib.get("bin_confidence", []), dtype=float)
            bin_count = np.array(calib.get("bin_count", []), dtype=int)
            ece = calib.get("ece", None)
            mask = bin_count > 0

            axes[1, 1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Ideal")
            axes[1, 1].plot(bin_conf[mask], bin_acc[mask], marker="o", label="Empirical")
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            title = "Calibration (Reliability)"
            if ece is not None:
                title += f"  |  ECE={float(ece):.3f}"
            axes[1, 1].set_title(title)
            axes[1, 1].set_xlabel("Mean confidence")
            axes[1, 1].set_ylabel("Accuracy")
            axes[1, 1].legend()
        else:
            axes[1, 1].axis("off")

        per_letter = results["detailed_analysis"]["per_letter_performance"]
        letters_perf = list(per_letter.keys())
        accuracies = [per_letter[letter]["accuracy"] for letter in letters_perf]
        axes[2, 0].bar(letters_perf, accuracies)
        axes[2, 0].set_title("Per-letter Accuracy")
        axes[2, 0].set_xlabel("Letter")
        axes[2, 0].set_ylabel("Accuracy")
        axes[2, 0].tick_params(axis="x", rotation=45)

        basic = results["basic_evaluation"]
        model_id = results.get("model_id") or "unknown"
        device = getattr(self.model, "device", "unknown")
        model_obj = getattr(self.model, "model", None)
        total_params = None
        trainable_params = None
        if model_obj is not None and hasattr(model_obj, "parameters"):
            try:
                params = list(model_obj.parameters())
                total_params = sum(p.numel() for p in params)
                trainable_params = sum(p.numel() for p in params if getattr(p, "requires_grad", False))
            except Exception:
                total_params = None
                trainable_params = None

        model_id_str = str(model_id)
        if len(model_id_str) > 72:
            model_id_str = "..." + model_id_str[-69:]

        lines: List[str] = []
        dataset_name = results.get("dataset_name")
        split_name = results.get("split")
        if dataset_name or split_name:
            lines.append(f"Dataset: {dataset_name or 'unknown'}  split={split_name or 'unknown'}")

        lines.extend(textwrap.wrap(f"Model ID: {model_id_str}", width=52) or [f"Model ID: {model_id_str}"])
        lines.append(f"Device: {device}")
        try:
            if str(device).startswith("cuda") and torch.cuda.is_available():
                lines.append(f"CUDA: {torch.cuda.get_device_name(0)}")
                try:
                    mem_alloc = torch.cuda.memory_allocated(0) / (1024**2)
                    mem_reserved = torch.cuda.memory_reserved(0) / (1024**2)
                    lines.append(f"CUDA mem (MB): alloc={mem_alloc:.0f}, reserved={mem_reserved:.0f}")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            mps_backend = getattr(torch.backends, "mps", None)
            if str(device) == "mps" and mps_backend is not None:
                lines.append(f"MPS built: {bool(getattr(mps_backend, 'is_built', lambda: False)())}")
                lines.append(f"MPS available: {bool(getattr(mps_backend, 'is_available', lambda: False)())}")
                try:
                    mps_mod = getattr(torch, "mps", None)
                    if mps_mod is not None and hasattr(mps_mod, "current_allocated_memory"):
                        mem_alloc = float(mps_mod.current_allocated_memory()) / (1024**2)
                        lines.append(f"MPS mem (MB): alloc={mem_alloc:.0f}")
                except Exception:
                    pass
        except Exception:
            pass
        if total_params is not None:
            lines.append(f"Params (total): {total_params:,}")
        if trainable_params is not None:
            lines.append(f"Params (trainable): {trainable_params:,}")

        best_meta = results.get("best_checkpoint_metadata") or {}
        if isinstance(best_meta, dict) and best_meta.get("best_checkpoint"):
            best_ckpt = str(best_meta.get("best_checkpoint"))
            best_metrics = best_meta.get("metrics") if isinstance(best_meta.get("metrics"), dict) else {}
            acc = best_metrics.get("accuracy")
            loss = best_metrics.get("loss")
            meta_line = f"Best ckpt: {best_ckpt}"
            if isinstance(acc, (int, float)):
                meta_line += f"  acc={float(acc):.3f}"
            if isinstance(loss, (int, float)):
                meta_line += f"  loss={float(loss):.3f}"
            lines.extend(textwrap.wrap(meta_line, width=60) or [meta_line])

        topk_k = basic.get("topk_k")
        topk_acc = basic.get("topk_accuracy")
        elapsed = basic.get("elapsed_seconds")
        sps = basic.get("samples_per_second")
        ece = (results.get("detailed_analysis") or {}).get("calibration", {}).get("ece")
        macro_f1 = (results.get("detailed_analysis") or {}).get("macro_f1")
        micro_f1 = (results.get("detailed_analysis") or {}).get("micro_f1")
        macro_p = (results.get("detailed_analysis") or {}).get("macro_precision")
        macro_r = (results.get("detailed_analysis") or {}).get("macro_recall")
        lines += [
            "",
            f"Overall Accuracy: {basic['accuracy']:.2%}",
            (f"Top-{topk_k} Accuracy: {float(topk_acc):.2%}" if topk_k and topk_acc is not None else "Top-k Accuracy: n/a"),
            f"Average Confidence: {basic['avg_confidence']:.3f}",
            f"Correct Predictions: {basic['correct_predictions']}/{basic['total_predictions']}",
        ]
        if elapsed is not None:
            lines.append(f"Runtime: {float(elapsed):.1f}s  ({float(sps or 0.0):.2f} samples/s)")
        if macro_f1 is not None and micro_f1 is not None:
            lines.append(f"F1 (macro/micro): {float(macro_f1):.3f} / {float(micro_f1):.3f}")
        if macro_p is not None and macro_r is not None:
            lines.append(f"P/R (macro): {float(macro_p):.3f} / {float(macro_r):.3f}")
        if ece is not None:
            lines.append(f"Calibration ECE: {float(ece):.3f}")

        counts = basic.get("per_class_counts") or {}
        if counts:
            values = list(counts.values())
            lines.append(f"Class coverage: min={min(values)}, max={max(values)}, classes={len(values)}")

        pairs = (results.get("detailed_analysis") or {}).get("most_confused_pairs") or []
        if pairs:
            lines.append("")
            lines.append("Most confused (true→pred):")
            for item in pairs[:6]:
                lines.append(f"  {item['count']}: {item['true']}→{item['pred']}")

        axes[2, 1].text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=axes[2, 1].transAxes,
            fontsize=10.5,
            ha="left",
            va="top",
            wrap=True,
            clip_on=True,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
        )
        axes[2, 1].set_title("Model + Performance", pad=10)
        axes[2, 1].axis("off")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        if show_seconds and show_seconds > 0:
            plt.show(block=False)
            plt.pause(show_seconds)
            plt.close(fig)
        else:
            plt.show()

    def export_confusion_matrix(
        self,
        results: Dict[str, Any],
        out_dir: str | Path,
        *,
        prefix: str = "",
        formats: Sequence[str] = ("csv", "json", "png"),
    ) -> Dict[str, str]:
        """
        Export the confusion matrix from a results dict to disk.

        Args:
            results: Dict returned by `evaluate_comprehensive`.
            out_dir: Output directory path.
            prefix: Optional filename prefix. When set, filenames become
                `{prefix}_confusion_matrix.<ext>`.
            formats: Iterable of formats to write: csv/json/png.

        Returns:
            Mapping of format -> written filepath.
        """
        cm = results["detailed_analysis"]["confusion_matrix"]
        stem = "confusion_matrix" if not prefix else f"{prefix}_confusion_matrix"
        return export_confusion_matrix(cm, self.model.letters, out_dir, stem=stem, formats=formats)

    def print_detailed_report(self, results: Dict[str, Any]):
        """
        Print a human-readable summary report to stdout.

        This is a convenience for CLI usage; it does not write files.
        """
        print(f"\n{'='*60}")
        print("DETAILED EVALUATION REPORT")
        print(f"{'='*60}")

        basic = results["basic_evaluation"]
        detailed = results["detailed_analysis"]

        print(f"\nModel: {results['model_name']}")
        print(f"Samples evaluated: {results['num_samples']}")
        print(f"Overall accuracy: {basic['accuracy']:.2%}")
        print(f"Average confidence: {basic['avg_confidence']:.3f}")

        print("\nConfidence Analysis:")
        conf_analysis = detailed["confidence_analysis"]
        print(f"  Mean confidence (all): {conf_analysis['mean_confidence']:.3f}")
        print(
            f"  Mean confidence (correct): {conf_analysis['mean_correct_confidence']:.3f}"
        )
        print(
            f"  Mean confidence (incorrect): {conf_analysis['mean_incorrect_confidence']:.3f}"
        )
        print(f"  Confidence std: {conf_analysis['confidence_std']:.3f}")

        print("\nPer-letter Performance:")
        per_letter = detailed["per_letter_performance"]
        for letter in sorted(per_letter.keys()):
            stats = per_letter[letter]
            print(
                f"  {letter}: {stats['accuracy']:.2%} ({stats['correct_predictions']}/{stats['total_samples']})"
            )

        print("\nDifficult Samples (top 5):")
        difficult = detailed["difficult_samples"][:5]
        for i, sample in enumerate(difficult):
            print(
                f"  {i+1}. True: {sample['true_letter']}, Pred: {sample['pred_letter']}, "
                f"Conf: {sample['confidence']:.3f}, Correct: {sample['correct']}"
            )


def export_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: Sequence[str],
    out_dir: str | Path,
    *,
    stem: str = "confusion_matrix",
    formats: Sequence[str] = ("csv", "json", "png"),
) -> Dict[str, str]:
    """
    Export a confusion matrix in one or more formats.

    - csv: header row is labels; first column is the true label
    - json: {"labels": [...], "matrix": [[...]]}
    - png: seaborn heatmap
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(confusion_matrix)
    labels = list(labels)
    exported: Dict[str, str] = {}

    normalized_formats = [f.strip().lower() for f in formats if f and f.strip()]
    if not normalized_formats:
        return exported

    if "csv" in normalized_formats:
        csv_path = out_path / f"{stem}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred", *labels])
            for i, true_label in enumerate(labels):
                writer.writerow([true_label, *matrix[i].tolist()])
        exported["csv"] = str(csv_path)

    if "json" in normalized_formats:
        json_path = out_path / f"{stem}.json"
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "labels": labels,
            "matrix": matrix.tolist(),
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        exported["json"] = str(json_path)

    if "png" in normalized_formats:
        png_path = out_path / f"{stem}.png"
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=False,
            fmt="g",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        exported["png"] = str(png_path)

    return exported


def parse_args() -> argparse.Namespace:
    """
    CLI argument parser for running the evaluator as a module.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Run ASL evaluator on a predictor.")
    parser.add_argument(
        "--model-id",
        help="checkpoint path or HF model id; defaults to ASL_MODEL_ID, then best_checkpoint.json, then base clip",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="device to run inference on: auto|cpu|mps|cuda (default: auto)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="log progress every N samples (0 disables)",
    )
    parser.add_argument(
        "--topk-k",
        type=int,
        default=5,
        help="compute top-k accuracy for this k (requires probs; default: 5)",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="number of bins for calibration/ECE (default: 10)",
    )
    parser.add_argument("--num-samples", type=int, default=50, help="number of samples to evaluate")
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="evaluate the entire dataset split (ignores --num-samples)",
    )
    parser.add_argument(
        "--per-class-samples",
        type=int,
        default=0,
        help="if set, evaluate up to N samples per class (label), stratified across the split",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed used for per-class sampling")
    parser.add_argument("--split", default="train", help="dataset split to use")
    parser.add_argument(
        "--export-dir",
        default="ml/artifacts/eval",
        help="directory to save artifacts (dashboard + confusion matrix exports)",
    )
    parser.add_argument(
        "--export-formats",
        default="csv,json,png",
        help="comma-separated formats for confusion matrix export",
    )
    parser.add_argument("--export-prefix", default="", help="filename prefix for exported artifacts")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="disable saving artifacts to disk (still shows the matplotlib window)",
    )
    parser.add_argument(
        "--show-seconds",
        type=float,
        default=30.0,
        help="seconds to show the matplotlib dashboard before auto-closing (0 = wait until you close it)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    model_id = (
        args.model_id
        or os.environ.get("ASL_MODEL_ID")
        or resolve_best_checkpoint_model_id(project_root=PROJECT_ROOT)
        or "openai/clip-vit-base-patch32"
    )
    device = None if args.device == "auto" else args.device
    log_torch_device_diagnostics(device)
    predictor = ASLPredictor(model_name=model_id, device=device)
    evaluator = ASLEvaluator(model=predictor)
    evaluator.log_every = max(0, int(args.log_every))
    evaluator.topk_metrics_k = max(1, int(args.topk_k))
    evaluator.ece_bins = max(2, int(args.ece_bins))
    logger.info("Predictor running on device: %s", getattr(predictor, "device", "unknown"))
    evaluator.load_dataset(split=args.split)
    indices = None
    if args.per_class_samples and args.per_class_samples > 0:
        indices = evaluator.select_indices(per_class_samples=args.per_class_samples, seed=args.seed)
    elif args.all_samples:
        indices = evaluator.select_indices(per_class_samples=None)

    results = evaluator.evaluate_comprehensive(num_samples=args.num_samples, indices=indices)
    results["dataset_name"] = evaluator.dataset_name
    results["split"] = evaluator.split
    results["best_checkpoint_metadata"] = load_best_checkpoint_metadata()
    basic = results["basic_evaluation"]
    print({"accuracy": basic["accuracy"], "avg_confidence": basic["avg_confidence"]})

    out_dir = args.export_dir
    prefix = args.export_prefix.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dashboard_path = None
    if not args.no_save and out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        dashboard_path = str(Path(out_dir) / f"{prefix}_dashboard.png")

    evaluator.visualize_results(results, save_path=dashboard_path, show_seconds=args.show_seconds)

    if not args.no_save and out_dir:
        formats = [f.strip() for f in args.export_formats.split(",") if f.strip()]
        exported = evaluator.export_confusion_matrix(
            results,
            out_dir,
            prefix=prefix,
            formats=formats,
        )
        print({"saved_dir": out_dir, "dashboard_png": dashboard_path, "confusion_matrix_exported": exported})
