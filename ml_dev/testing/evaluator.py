"""
Comprehensive evaluation framework for ASL models
"""

from collections import defaultdict
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset


class ASLEvaluator:
    """Comprehensive evaluator for ASL models"""

    def __init__(
        self,
        model,
        dataset_name: str = "aliciiavs/sign_language_image_dataset",
    ):
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = None
        self.split = None

    def load_dataset(self, split: str = "train"):
        """Load dataset for evaluation"""
        print(f"Loading {self.dataset_name} dataset...")
        self.dataset = load_dataset(self.dataset_name)
        self.split = split
        print(f"Loaded {len(self.dataset[split])} samples from {split} split")

    def evaluate_comprehensive(self, num_samples: int = 100) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        if self.dataset is None:
            self.load_dataset()

        # Ensure split is set
        if not hasattr(self, "split"):
            self.split = "train"

        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION - {self.model.__class__.__name__}")
        print(f"{'='*60}")

        # Basic evaluation
        basic_results = self._evaluate_basic(num_samples)

        # Detailed analysis
        detailed_results = self._analyze_detailed(num_samples)

        # Combine results
        results = {
            "model_name": self.model.__class__.__name__,
            "basic_evaluation": basic_results,
            "detailed_analysis": detailed_results,
            "num_samples": num_samples,
        }

        return results

    def _evaluate_basic(self, num_samples: int) -> Dict[str, Any]:
        """Basic accuracy evaluation"""
        correct = 0
        confidences = []
        all_predictions = []

        print(f"Running basic evaluation on {num_samples} samples...")

        for i in range(min(num_samples, len(self.dataset[self.split]))):
            sample = self.dataset[self.split][i]
            image = sample["image"]
            true_label = sample["label"]
            true_letter = self.model.letters[true_label]

            # Get prediction
            pred_letter, confidence, _ = self.model.predict(image)

            if pred_letter == true_letter:
                correct += 1

            confidences.append(confidence)
            all_predictions.append(
                {
                    "true_letter": true_letter,
                    "pred_letter": pred_letter,
                    "confidence": confidence,
                    "correct": pred_letter == true_letter,
                    "sample_index": i,
                }
            )

            if i < 3:  # Print first few examples
                print(
                    f"Sample {i}: True={true_letter}, Pred={pred_letter}, Conf={confidence:.3f}"
                )

        accuracy = correct / num_samples
        avg_confidence = np.mean(confidences)

        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "correct_predictions": correct,
            "total_predictions": num_samples,
            "predictions": all_predictions,
        }

    def _analyze_detailed(self, num_samples: int) -> Dict[str, Any]:
        """Detailed analysis of model performance"""
        print("Running detailed analysis...")

        # Confusion matrix
        confusion_matrix = self._compute_confusion_matrix(num_samples)

        # Confidence analysis
        confidence_analysis = self._analyze_confidence_distribution(num_samples)

        # Per-letter performance
        per_letter_performance = self._analyze_per_letter_performance(num_samples)

        # Difficult samples
        difficult_samples = self._find_difficult_samples(num_samples)

        return {
            "confusion_matrix": confusion_matrix,
            "confidence_analysis": confidence_analysis,
            "per_letter_performance": per_letter_performance,
            "difficult_samples": difficult_samples,
        }

    def _compute_confusion_matrix(self, num_samples: int) -> np.ndarray:
        """Compute confusion matrix"""
        confusion = defaultdict(lambda: defaultdict(int))

        for i in range(min(num_samples, len(self.dataset[self.split]))):
            sample = self.dataset[self.split][i]
            image = sample["image"]
            true_label = sample["label"]
            true_letter = self.model.letters[true_label]

            pred_letter, _, _ = self.model.predict(image)
            confusion[true_letter][pred_letter] += 1

        # Convert to matrix
        matrix = np.zeros((len(self.model.letters), len(self.model.letters)))
        for i, true_letter in enumerate(self.model.letters):
            for j, pred_letter in enumerate(self.model.letters):
                matrix[i, j] = confusion[true_letter][pred_letter]

        return matrix

    def _analyze_confidence_distribution(self, num_samples: int) -> Dict[str, Any]:
        """Analyze confidence score distribution"""
        confidences = []
        correct_confidences = []
        incorrect_confidences = []

        for i in range(min(num_samples, len(self.dataset[self.split]))):
            sample = self.dataset[self.split][i]
            image = sample["image"]
            true_label = sample["label"]
            true_letter = self.model.letters[true_label]

            pred_letter, confidence, _ = self.model.predict(image)

            confidences.append(confidence)

            if pred_letter == true_letter:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)

        return {
            "all_confidences": confidences,
            "correct_confidences": correct_confidences,
            "incorrect_confidences": incorrect_confidences,
            "mean_confidence": np.mean(confidences),
            "mean_correct_confidence": (
                np.mean(correct_confidences) if correct_confidences else 0
            ),
            "mean_incorrect_confidence": (
                np.mean(incorrect_confidences) if incorrect_confidences else 0
            ),
            "confidence_std": np.std(confidences),
        }

    def _analyze_per_letter_performance(
        self, num_samples: int
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze performance for each letter"""
        letter_stats = defaultdict(
            lambda: {"correct": 0, "total": 0, "confidences": []}
        )

        for i in range(min(num_samples, len(self.dataset[self.split]))):
            sample = self.dataset[self.split][i]
            image = sample["image"]
            true_label = sample["label"]
            true_letter = self.model.letters[true_label]

            pred_letter, confidence, _ = self.model.predict(image)

            letter_stats[true_letter]["total"] += 1
            letter_stats[true_letter]["confidences"].append(confidence)

            if pred_letter == true_letter:
                letter_stats[true_letter]["correct"] += 1

        # Calculate per-letter metrics
        per_letter_results = {}
        for letter in self.model.letters:
            if letter in letter_stats:
                stats = letter_stats[letter]
                accuracy = (
                    stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                )
                avg_confidence = (
                    np.mean(stats["confidences"]) if stats["confidences"] else 0
                )

                per_letter_results[letter] = {
                    "accuracy": accuracy,
                    "avg_confidence": avg_confidence,
                    "total_samples": stats["total"],
                    "correct_predictions": stats["correct"],
                }

        return per_letter_results

    def _find_difficult_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Find samples that are difficult to classify"""
        difficult_samples = []

        for i in range(min(num_samples, len(self.dataset[self.split]))):
            sample = self.dataset[self.split][i]
            image = sample["image"]
            true_label = sample["label"]
            true_letter = self.model.letters[true_label]

            pred_letter, confidence, probs = self.model.predict(image)

            # Consider difficult if low confidence or incorrect
            if confidence < 0.5 or pred_letter != true_letter:
                difficult_samples.append(
                    {
                        "index": i,
                        "true_letter": true_letter,
                        "pred_letter": pred_letter,
                        "confidence": confidence,
                        "correct": pred_letter == true_letter,
                        "top3_probs": torch.topk(probs[0], 3),
                    }
                )

        # Sort by confidence (lowest first)
        difficult_samples.sort(key=lambda x: x["confidence"])

        return difficult_samples[:10]  # Return top 10 most difficult

    def visualize_results(
        self, results: Dict[str, Any], save_path: Optional[str] = None
    ):
        """Visualize evaluation results"""
        print("Creating visualizations...")

        # Create subplots
        _, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Confusion Matrix
        confusion_matrix = results["detailed_analysis"]["confusion_matrix"]
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=self.model.letters,
            yticklabels=self.model.letters,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("True")

        # 2. Confidence Distribution
        confidence_analysis = results["detailed_analysis"]["confidence_analysis"]
        axes[0, 1].hist(
            confidence_analysis["all_confidences"], bins=20, alpha=0.7, label="All"
        )
        axes[0, 1].hist(
            confidence_analysis["correct_confidences"],
            bins=20,
            alpha=0.7,
            label="Correct",
        )
        axes[0, 1].hist(
            confidence_analysis["incorrect_confidences"],
            bins=20,
            alpha=0.7,
            label="Incorrect",
        )
        axes[0, 1].set_title("Confidence Distribution")
        axes[0, 1].set_xlabel("Confidence Score")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()

        # 3. Per-letter Accuracy
        per_letter = results["detailed_analysis"]["per_letter_performance"]
        letters = list(per_letter.keys())
        accuracies = [per_letter[letter]["accuracy"] for letter in letters]

        axes[1, 0].bar(letters, accuracies)
        axes[1, 0].set_title("Per-letter Accuracy")
        axes[1, 0].set_xlabel("Letter")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # 4. Performance Summary
        basic = results["basic_evaluation"]
        axes[1, 1].text(
            0.1,
            0.8,
            f"Overall Accuracy: {basic['accuracy']:.2%}",
            transform=axes[1, 1].transAxes,
            fontsize=12,
        )
        axes[1, 1].text(
            0.1,
            0.7,
            f"Average Confidence: {basic['avg_confidence']:.3f}",
            transform=axes[1, 1].transAxes,
            fontsize=12,
        )
        axes[1, 1].text(
            0.1,
            0.6,
            f"Correct Predictions: {basic['correct_predictions']}/{basic['total_predictions']}",
            transform=axes[1, 1].transAxes,
            fontsize=12,
        )
        axes[1, 1].set_title("Performance Summary")
        axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        plt.show()

    def print_detailed_report(self, results: Dict[str, Any]):
        """Print detailed evaluation report"""
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


class ModelComparison:
    """Compare multiple ASL models"""

    def __init__(
        self,
        models,
        dataset_name: str = "aliciiavs/sign_language_image_dataset",
    ):
        self.models = models
        self.dataset_name = dataset_name
        self.dataset = None
        self.split = None

    def load_dataset(self, split: str = "train"):
        """Load dataset for comparison"""
        print(f"Loading {self.dataset_name} dataset...")
        self.dataset = load_dataset(self.dataset_name)
        self.split = split
        print(f"Loaded {len(self.dataset[split])} samples from {split} split")

    def compare_models(self, num_samples: int = 100) -> Dict[str, Any]:
        """Compare all models"""
        if self.dataset is None:
            self.load_dataset()

        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")

        results = {}

        for model in self.models:
            print(f"\nEvaluating {model.__class__.__name__}...")
            evaluator = ASLEvaluator(model, self.dataset_name)
            evaluator.dataset = self.dataset
            model_results = evaluator.evaluate_comprehensive(num_samples)
            results[model.__class__.__name__] = model_results

        # Create comparison summary
        comparison_summary = self._create_comparison_summary(results)
        results["comparison"] = comparison_summary

        return results

    def _create_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparison summary"""
        summary = {
            "models": [],
            "accuracies": {},
            "confidences": {},
            "best_accuracy": None,
            "highest_confidence": None,
        }

        best_accuracy = 0
        highest_confidence = 0

        for model_name, model_results in results.items():
            if model_name == "comparison":
                continue

            basic = model_results["basic_evaluation"]
            accuracy = basic["accuracy"]
            confidence = basic["avg_confidence"]

            summary["models"].append(model_name)
            summary["accuracies"][model_name] = accuracy
            summary["confidences"][model_name] = confidence

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                summary["best_accuracy"] = model_name

            if confidence > highest_confidence:
                highest_confidence = confidence
                summary["highest_confidence"] = model_name

        return summary

    def print_comparison_report(self, results: Dict[str, Any]):
        """Print comparison report"""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON REPORT")
        print(f"{'='*60}")

        comparison = results["comparison"]

        print("\nAccuracy Comparison:")
        for model_name in comparison["models"]:
            accuracy = comparison["accuracies"][model_name]
            print(f"  {model_name}: {accuracy:.2%}")

        print("\nConfidence Comparison:")
        for model_name in comparison["models"]:
            confidence = comparison["confidences"][model_name]
            print(f"  {model_name}: {confidence:.3f}")

        print(f"\nBest Accuracy: {comparison['best_accuracy']}")
        print(f"Highest Confidence: {comparison['highest_confidence']}")
