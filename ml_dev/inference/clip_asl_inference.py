"""
asl predictor built on clip

wraps a base or fine tuned checkpoint with asl prompts/preprocessing; used by:
- inference service (ml_dev.inference.service.get_predictor / django)
- testing/evaluator (ml_dev.testing.evaluator)

run (from repo root):
- set ASL_MODEL_ID to a fine-tuned checkpoint (e.g. ml_dev/saved_weights/epoch_7) or pass --model-id
- python -m ml_dev.inference.clip_asl_inference --model-id ml_dev/saved_weights/epoch_7 --image path/to/image.jpg --top-k 3
"""

import os
import sys
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# add the parent directory to path to import from development
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from development.clip_base import BaseASLModel
from development.preprocessing import ASLPreprocessor


class ASLPredictor(BaseASLModel):
    """ASL letter recognition using CLIP model."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        """
        Initialize the ASL predictor.

        Args:
            model_name: model id to load (hf repo id or local checkpoint); defaults to base clip when service does not set ASL_MODEL_ID.
            device: optional torch device string. defaults to "cuda" if available.
        """
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(target_device)

        print(f"Loading CLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name)
        
        # Try to move model to device, fall back to CPU if CUDA kernel error occurs
        try:
            self.model.to(self.device)
            # Test if CUDA actually works by doing a small operation
            if self.device.type == "cuda":
                test_tensor = torch.zeros(1, device=self.device)
                _ = test_tensor + 1  # Simple operation to trigger kernel execution
                print(f"CUDA compatibility check passed on {self.device}")
        except RuntimeError as e:
            if "CUDA error" in str(e) and "no kernel image is available" in str(e):
                print(f"Warning: CUDA kernel not compatible with device. Falling back to CPU.")
                print(f"Error details: {e}")
                self.device = torch.device("cpu")
                self.model.to(self.device)
            else:
                raise
        
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # set up letters and text prompts
        self.letters: List[str] = [chr(65 + i) for i in range(26)]  # a to z
        self.text_prompts = [
            f"a photo of a hand showing the sign language letter {letter}"
            for letter in self.letters
        ]

        # prepare text inputs once for reuse
        text_inputs = self.processor(
            text=self.text_prompts, return_tensors="pt", padding=True
        )
        self.text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # initialize preprocessor
        self.preprocessor = ASLPreprocessor()

        print(f"ASL Predictor initialized with {len(self.letters)} letters on {self.device}")

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Predict ASL letter from an image.

        Args:
            image: PIL Image to predict


        Returns:
            Tuple of (predicted_letter, confidence_score)
        """
        processed_image = self.preprocessor.preprocess(image)

        inputs = self.processor(
            images=processed_image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, **self.text_inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        pred_idx = probs.argmax().item()
        confidence = probs.max().item()
        predicted_letter = self.letters[pred_idx]

        return predicted_letter, confidence

    def predict_from_path(self, image_path: str) -> Tuple[str, float]:
        """
        Predict ASL letter from an image file path

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (predicted_letter, confidence_score)
        """
        try:
            image = Image.open(image_path)
            return self.predict(image)
        except Exception as e:
            raise ValueError(f"Error loading image from {image_path}: {e}") from e

    def predict_from_dataset(
        self, index: int = 0, split: str = "train"
    ) -> Tuple[str, float, str]:
        """
        Predict ASL letter from ASL dataset

        Args:
            index: Index of image in dataset
            split: Dataset split to use (train/test)

        Returns:
            Tuple of (predicted_letter, confidence_score, true_letter)
        """
        # load dataset cached after first load
        if not hasattr(self, "dataset"):
            print("Loading ASL dataset...")
            self.dataset = load_dataset("aliciiavs/sign_language_image_dataset")
            print(f"Dataset loaded with {len(self.dataset[split])} samples")

        # get image and true label
        sample = self.dataset[split][index]
        image = sample["image"]
        true_label = sample["label"]
        true_letter = self.letters[true_label]

        # predict
        predicted_letter, confidence = self.predict(image)

        return predicted_letter, confidence, true_letter

    def get_top_predictions(self, image: Image.Image, top_k: int = 3) -> list:
        """
        Get top-k predictions with confidence scores

        Args:
            image: PIL Image to predict
            top_k: Number of top predictions to return

        Returns:
            List of tuples (letter, confidence) sorted by confidence
        """
        # preprocess the image
        processed_image = self.preprocessor.preprocess(image)

        inputs = self.processor(
            images=processed_image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, **self.text_inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        top_probs, top_indices = torch.topk(probs[0], top_k)

        top_predictions = []
        for i in range(top_k):
            letter = self.letters[top_indices[i].item()]
            confidence = top_probs[i].item()
            top_predictions.append((letter, confidence))

        return top_predictions


# example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="run a quick ASL prediction")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ASL_MODEL_ID", "openai/clip-vit-base-patch32"),
        help="hf repo id or local checkpoint; defaults to ASL_MODEL_ID or base clip",
    )
    parser.add_argument("--image", required=True, help="path to image file")
    parser.add_argument(
        "--top-k", type=int, default=3, help="number of top predictions to print"
    )
    args = parser.parse_args()

    predictor = ASLPredictor(model_name=args.model_id)
    try:
        img = Image.open(args.image)
    except Exception as exc:
        raise SystemExit(f"failed to load image {args.image}: {exc}")

    letter, conf = predictor.predict(img)
    print(f"predicted: {letter} (conf {conf:.3f})")

    if args.top_k and args.top_k > 1:
        tops = predictor.get_top_predictions(img, top_k=args.top_k)
        print("top predictions:")
        for rank, (ltr, c) in enumerate(tops, start=1):
            print(f"  {rank}. {ltr}: {c:.3f}")
