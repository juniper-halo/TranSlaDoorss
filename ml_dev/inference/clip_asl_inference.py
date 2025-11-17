"""
ASL Letter Recognition using CLIP
"""
from typing import Tuple
import sys
import os
import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from PIL import Image


# Add the parent directory to path to import from development
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from development.preprocessing import ASLPreprocessor # pylint: disable=wrong-import-position


class ASLPredictor:
    """ASL letter recognition using CLIP model"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the ASL predictor
        
        Args:
            model_name: CLIP model name to use
        """
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Setup letters and text prompts
        self.letters = [chr(65 + i) for i in range(26)]  # A-Z
        self.text_prompts = [
            f"a photo of a hand showing the sign language letter {letter}"
            for letter in self.letters
        ]

        self.dataset = None

        # Initialize preprocessor
        self.preprocessor = ASLPreprocessor()

        print(f"ASL Predictor initialized with {len(self.letters)} letters")

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Predict ASL letter from an image
        
        Args:
            image: PIL Image to predict
            
        Returns:
            Tuple of (predicted_letter, confidence_score)
        """
        # Preprocess the image
        processed_image = self.preprocessor.preprocess(image)

        # Run CLIP inference
        inputs = self.processor(
            text=self.text_prompts,
            images=processed_image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        # Get prediction
        pred_idx = probs.argmax().item()
        confidence2 = probs.max().item()
        predicted_letter = self.letters[pred_idx]

        return predicted_letter, confidence2

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

    def predict_from_dataset(self, index: int = 0, split: str = "train") -> Tuple[str, float, str]:
        """
        Predict ASL letter from ASL dataset
        
        Args:
            index: Index of image in dataset
            split: Dataset split to use (train/test)
            
        Returns:
            Tuple of (predicted_letter, confidence_score, true_letter)
        """
        # Load dataset (cached after first load)
        if not hasattr(self, 'dataset'):
            print("Loading ASL dataset...")
            self.dataset = load_dataset("aliciiavs/sign_language_image_dataset")
            print(f"Dataset loaded with {len(self.dataset[split])} samples")

        # Get image and true label
        sample = self.dataset[split][index]
        image = sample['image']
        true_label = sample['label']
        true_letter2 = self.letters[true_label]

        # Predict
        predicted_letter, confidence3 = self.predict(image)

        return predicted_letter, confidence3, true_letter2

    def get_top_predictions(self, image: Image.Image, top_k: int = 3) -> list:
        """
        Get top-k predictions with confidence scores
        
        Args:
            image: PIL Image to predict
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (letter, confidence) sorted by confidence
        """
        # Preprocess the image
        processed_image = self.preprocessor.preprocess(image)

        # Run CLIP inference
        inputs = self.processor(
            text=self.text_prompts,
            images=processed_image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], top_k)

        top_predictions2 = []
        for k in range(top_k):
            letter2 = self.letters[top_indices[k].item()]
            confidence = top_probs[k].item()
            top_predictions2.append((letter2, confidence))

        return top_predictions2


# Example usage and testing
if __name__ == "__main__":
    print("Testing ASL Predictor...")

    # Initialize predictor
    predictor = ASLPredictor()

    # Test with dataset
    print("\nTesting with ASL dataset (index 0):")
    try:
        pred_letter, confidence4, true_letter = predictor.predict_from_dataset(index=0)
        print(f"True letter: {true_letter}")
        print(f"Predicted letter: {pred_letter}")
        print(f"Confidence: {confidence4:.3f}")
        print(f"Correct: {pred_letter == true_letter}")

        # Get top 3 predictions
        print("\nTop 3 predictions:")
        top_predictions = predictor.get_top_predictions(
            predictor.dataset['train'][0]['image'], top_k=3
        )
        for i, (letter, conf) in enumerate(top_predictions):
            print(f"  {i+1}. {letter}: {conf:.3f}")

    except Exception as e:
        print(f"Error testing with dataset: {e}")

    print("\nASL Predictor test completed!")
