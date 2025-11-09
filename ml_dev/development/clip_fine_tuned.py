"""
CLIP fine-tuning structure for ASL letter recognition.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import CLIPModel, CLIPProcessor, get_linear_schedule_with_warmup

# ensure project root is on sys.path when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml_dev.development.preprocessing import ASLPreprocessor


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
    output_dir: str = "artifacts/clip_finetune"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    log_every_n_steps: int = 25
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
        from datasets import load_dataset  # lazy import avoids CLI penalty

        self.dataset_cfg = dataset_cfg
        self.training_cfg = training_cfg

        self.processor = CLIPProcessor.from_pretrained(training_cfg.model_name)
        self.model = CLIPModel.from_pretrained(training_cfg.model_name)
        self.model.to(training_cfg.device)

        self.preprocessor = ASLPreprocessor()

        # Prepare textual prompts and precompute text features (kept fixed here?)
        self.letters = [chr(65 + i) for i in range(26)]
        self.text_prompts = [
            f"a photo of a hand showing the sign language letter {letter}"
            for letter in self.letters
        ]

        # tokenize and compute text features 
        tokenized = self.processor(text=self.text_prompts, return_tensors="pt", padding=True)
        tokenized = {k: v.to(training_cfg.device) for k, v in tokenized.items()}
        with torch.no_grad():
            text_feats = self.model.get_text_features(**tokenized)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        self.register_buffer = None  # placeholder
        self.text_features = text_feats

        # loss
        self.criterion = nn.CrossEntropyLoss()

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

    def _create_optimizer(self) -> AdamW:
        params = [
            {"params": self.model.vision_model.parameters(), "lr": self.training_cfg.learning_rate},
        ]
        return AdamW(params, lr=self.training_cfg.learning_rate, weight_decay=self.training_cfg.weight_decay)

    def _create_scheduler(self):
        total_steps = (
            len(self.train_dataset)
            // self.training_cfg.batch_size
            // self.training_cfg.gradient_accumulation_steps
            * self.training_cfg.epochs
        )
        total_steps = max(1, total_steps)
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
        val_loader = self._create_dataloader(self.val_dataset, shuffle=False) if len(self.val_dataset) > 0 else None
        global_step = 0

        self.model.train()

        for epoch in range(self.training_cfg.epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(train_loader):
                # batch = {k: v.to(self.training_cfg.device) for k, v in batch.items()}
                # outputs = self.model(**batch)
                # loss = outputs.logits_per_image.softmax(dim=1)  # placeholder

                # # TODO: replace with cross-entropy against labels
                # loss = torch.tensor(0.0, device=self.training_cfg.device)

                # dataset returns 'pixel_values' and 'labels'
                pixel_values = batch["pixel_values"].to(self.training_cfg.device)
                labels = batch["labels"].to(self.training_cfg.device)

                # get image features
                image_feats = self.model.get_image_features(pixel_values=pixel_values)
                image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
                text_feats = self.text_features.to(self.training_cfg.device)

                # compute logits: scaled cosine similarity
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * image_feats @ text_feats.t()

                # cross-entropy: ground-truth label indices
                loss = self.criterion(logits_per_image, labels)
                loss.backward()

                epoch_loss += loss.item()
                epoch_steps += 1

                if (step + 1) % self.training_cfg.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % self.training_cfg.log_every_n_steps == 0:
                        log.info("Epoch %s step %s placeholder loss %.4f", epoch, global_step, loss.item())

                        avg_loss = epoch_loss / max(1, epoch_steps)
                        log.info("Epoch %s step %s loss %.4f", epoch, global_step, avg_loss)

            avg_epoch_loss = epoch_loss / max(1, epoch_steps)
            log.info("Finished epoch %s - avg loss: %.4f", epoch, avg_epoch_loss)

            if val_loader:
                self.evaluate(val_loader, epoch)

            self.save_checkpoint(epoch)

    def evaluate(self, dataloader: DataLoader, epoch: int) -> None:
        self.model.eval()
        # TODO: add model evaluation logic (accuracy, confusion matrix, etc.)
        log.info("Ran evaluation epoch %s (placeholder)", epoch)
        self.model.train()

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP for ASL recognition.")
    parser.add_argument("--config", type=str, help="Optional JSON config file with dataset/training settings.")
    parser.add_argument("--output-dir", type=str, help="Override output directory for checkpoints.")
    return parser.parse_args()


def load_configs(config_path: Optional[str], output_override: Optional[str]) -> Tuple[DatasetConfig, TrainingConfig]:
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


if __name__ == "__main__":
    main()