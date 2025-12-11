"""
clip fine tuning structure for asl letter recognition

trains a clip vision tower on the asl dataset and saves checkpoints plus metrics files
that can be selected later with export_best_checkpoint.py and loaded via the inference service
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPModel, CLIPProcessor, get_linear_schedule_with_warmup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ensure project root is on sys path when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml_dev.development.preprocessing import \
    ASLPreprocessor  # pylint: disable=wrong-import-position

log = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetConfig:
    name: str = "aliciiavs/sign_language_image_dataset"
    train_split: str = "train"
    validation_ratio: float = 0.1
    max_train_samples: Optional[int] = None
    seed: int = 42


@dataclasses.dataclass
class TrainingConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    output_dir: str = "saved_weights"
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    log_every_n_steps: int = 25
    log_dir: Optional[str] = None
    log_tensorboard: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ASLDataset(Dataset):
    """Torch dataset wrapper around the ASL Hugging Face dataset."""

    def __init__(
        self,
        hf_dataset: Iterable[Dict[str, Any]],
        preprocess: ASLPreprocessor,
        processor: CLIPProcessor,
    ) -> None:
        self.dataset = hf_dataset
        self.preprocess = preprocess
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        image = self.preprocess.preprocess(sample["image"])
        inputs = self.processor(images=image, return_tensors="pt")
        item = {key: value.squeeze(0) for key, value in inputs.items()}
        item["labels"] = torch.tensor(sample["label"], dtype=torch.long)
        return item


class ASLFineTuner:
    """High-level trainer that orchestrates CLIP fine-tuning."""

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        training_cfg: TrainingConfig,
    ) -> None:
        from datasets import load_dataset  # lazy import avoids cli penalty

        self.dataset_cfg = dataset_cfg
        self.training_cfg = training_cfg

        self.processor = CLIPProcessor.from_pretrained(training_cfg.model_name)
        self.model = CLIPModel.from_pretrained(training_cfg.model_name)
        self.model.to(training_cfg.device)

        self.preprocessor = ASLPreprocessor()
        self.letters = [chr(65 + i) for i in range(26)]
        self.text_prompts = [f"a photo of a hand showing the sign language letter {letter}" for letter in self.letters]
        text_inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True)
        self.text_inputs = {k: v.to(training_cfg.device) for k, v in text_inputs.items()}
        self.loss_fn = nn.CrossEntropyLoss()

        hf_dataset = load_dataset(dataset_cfg.name)
        train_split = hf_dataset[dataset_cfg.train_split]

        if dataset_cfg.max_train_samples:
            train_split = train_split.select(range(dataset_cfg.max_train_samples))

        val_size = int(len(train_split) * dataset_cfg.validation_ratio)
        train_size = len(train_split) - val_size
        self.train_dataset, self.val_dataset = random_split(
            ASLDataset(train_split, self.preprocessor, self.processor),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(dataset_cfg.seed),
        )

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # logging
        log_dir = Path(training_cfg.log_dir) if training_cfg.log_dir else Path(training_cfg.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir) if training_cfg.log_tensorboard else None
        if self.writer:
            log.info("TensorBoard logging to %s", log_dir.resolve())

    def _create_optimizer(self) -> AdamW:
        params = [
            {
                "params": self.model.vision_model.parameters(),
                "lr": self.training_cfg.learning_rate,
            },
        ]
        return AdamW(
            params,
            lr=self.training_cfg.learning_rate,
            weight_decay=self.training_cfg.weight_decay,
        )

    def _create_scheduler(self):
        steps_per_epoch = math.ceil(len(self.train_dataset) / self.training_cfg.batch_size)
        total_steps = max(
            1, steps_per_epoch * self.training_cfg.epochs // self.training_cfg.gradient_accumulation_steps
        )
        return get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.training_cfg.warmup_steps,
            num_training_steps=total_steps,
        )

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.training_cfg.batch_size,
            shuffle=shuffle,
            num_workers=os.cpu_count() or 1,
        )

    def train(self) -> None:
        train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        val_loader = (
            self._create_dataloader(self.val_dataset, shuffle=False)
            if len(self.val_dataset) > 0
            else None
        )
        global_step = 0

        self.model.train()

        for epoch in range(self.training_cfg.epochs):
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.training_cfg.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = self.model(**batch, **self.text_inputs)
                logits = outputs.logits_per_image
                raw_loss = self.loss_fn(logits, labels)
                loss = raw_loss / self.training_cfg.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.training_cfg.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    batch_pred = logits.argmax(dim=1)
                    running_total += labels.size(0)
                    running_correct += (batch_pred == labels).sum().item()
                    running_loss += raw_loss.item()

                    if global_step % self.training_cfg.log_every_n_steps == 0:
                        avg_loss = running_loss / max(1, running_total)
                        acc = running_correct / max(1, running_total)
                        log.info("Epoch %s step %s | loss %.4f | acc %.3f", epoch, global_step, avg_loss, acc)
                        if self.writer:
                            self.writer.add_scalar("train/loss", avg_loss, global_step)
                            self.writer.add_scalar("train/accuracy", acc, global_step)

            if val_loader:
                eval_metrics = self.evaluate(val_loader, epoch)
                if self.writer:
                    self.writer.add_scalar("val/loss", eval_metrics["loss"], epoch)
                    self.writer.add_scalar("val/accuracy", eval_metrics["accuracy"], epoch)
                self._write_metrics(eval_metrics, epoch)

            self.save_checkpoint(epoch)

    def evaluate(self, dataloader: DataLoader, epoch: int) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        confusion = torch.zeros(len(self.letters), len(self.letters), dtype=torch.long)

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.training_cfg.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = self.model(**batch, **self.text_inputs)
                logits = outputs.logits_per_image
                loss = self.loss_fn(logits, labels)

                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item() * labels.size(0)
                for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
                    confusion[true_label.long(), pred_label.long()] += 1

        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        per_class_total = confusion.sum(dim=1)
        per_class_correct = confusion.diag()
        per_class_accuracy = torch.where(per_class_total > 0, per_class_correct.float() / per_class_total.float(), torch.zeros_like(per_class_total, dtype=torch.float))

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "per_class_accuracy": {letter: per_class_accuracy[idx].item() for idx, letter in enumerate(self.letters)},
        }

        log.info("Validation epoch %s | loss %.4f | accuracy %.3f", epoch, avg_loss, accuracy)
        log.debug("Per-class accuracy: %s", metrics["per_class_accuracy"])
        self.model.train()
        return metrics

    def save_checkpoint(self, epoch: int) -> None:
        output_dir = Path(self.training_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / f"epoch_{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)

        cfg_path = checkpoint_dir / "config.json"
        with cfg_path.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "dataset": dataclasses.asdict(self.dataset_cfg),
                    "training": dataclasses.asdict(self.training_cfg),
                },
                fp,
                indent=2,
            )

        log.info("Saved checkpoint to %s", checkpoint_dir)
        if self.writer:
            self.writer.flush()

    def _write_metrics(self, metrics: Dict[str, Any], epoch: int) -> None:
        """Persist evaluation metrics to disk for later comparison."""
        output_dir = Path(self.training_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / f"metrics_epoch_{epoch}.json"
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        log.info("Saved metrics to %s", metrics_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP for ASL recognition.")
    parser.add_argument(
        "--config",
        type=str,
        help="Optional JSON config file with dataset/training settings.",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Override output directory for checkpoints."
    )
    return parser.parse_args()


def load_configs(
    config_path: Optional[str], output_override: Optional[str]
) -> Tuple[DatasetConfig, TrainingConfig]:
    dataset_cfg = DatasetConfig()
    training_cfg = TrainingConfig()

    if config_path:
        with open(config_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if "dataset" in payload:
            dataset_cfg = DatasetConfig(**payload["dataset"])
        if "training" in payload:
            training_cfg = TrainingConfig(**payload["training"])

    if output_override:
        training_cfg.output_dir = output_override

    return dataset_cfg, training_cfg


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    dataset_cfg, training_cfg = load_configs(args.config, args.output_dir)
    trainer = ASLFineTuner(dataset_cfg, training_cfg)
    trainer.train()
    log.info("training complete; run export_best_checkpoint.py to pick a best checkpoint based on metrics")


if __name__ == "__main__":
    main()
