"""
main Entry Point
"""

import argparse
import os
import sys

from PIL import Image

# add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import our ASL predictor
from inference.clip_asl_inference import \
    ASLPredictor  # pylint: disable=wrong-import-position


def main():
    """Main function for ASL letter recognition"""
    parser = argparse.ArgumentParser(description="ASL Letter Recognition using CLIP")
    parser.add_argument("--image", type=str, help="Path to image file to predict")
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="Index of image in ASL dataset (default: 0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions to show (default: 1)",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("ASL Letter Recognition")
    print("=" * 50)

    # Initialize ASL predictor
    print("Loading ASL predictor...")
    try:
        predictor = ASLPredictor()
        print("✓ ASL predictor loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading ASL predictor: {e}")
        return

    # Determine which image to use
    if args.image:
        # Use custom image file
        print(f"\nPredicting from image file: {args.image}")
        try:
            if args.top_k == 1:
                # Single prediction
                predicted_letter, confidence = predictor.predict_from_path(args.image)
                print(f"\nPredicted Letter: {predicted_letter}")
                print(f"Confidence: {confidence:.3f}")
            else:
                # Top-k predictions
                image = Image.open(args.image)
                top_predictions = predictor.get_top_predictions(image, top_k=args.top_k)
                print(f"\nTop {args.top_k} predictions:")
                for i, (letter, conf) in enumerate(top_predictions):
                    print(f"  {i+1}. {letter}: {conf:.3f}")

        except Exception as e:
            print(f"✗ Error processing image: {e}")
            return

    else:
        # Use dataset image
        print(f"\nPredicting from ASL dataset (index {args.dataset_index})")
        try:
            if args.top_k == 1:
                # Single prediction
                predicted_letter, confidence, true_letter = (
                    predictor.predict_from_dataset(index=args.dataset_index)
                )
                print(f"\nTrue Letter: {true_letter}")
                print(f"Predicted Letter: {predicted_letter}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Correct: {'✓' if predicted_letter == true_letter else '✗'}")
            else:
                # Top-k predictions
                sample = predictor.dataset["train"][args.dataset_index]
                image = sample["image"]
                true_label = sample["label"]
                true_letter = predictor.letters[true_label]

                top_predictions = predictor.get_top_predictions(image, top_k=args.top_k)
                print(f"\nTrue Letter: {true_letter}")
                print(f"Top {args.top_k} predictions:")
                for i, (letter, conf) in enumerate(top_predictions):
                    marker = "✓" if letter == true_letter else " "
                    print(f"  {i+1}. {letter}: {conf:.3f} {marker}")

        except Exception as e:
            print(f"✗ Error processing dataset image: {e}")
            return

    print("\n" + "=" * 50)
    print("ASL Recognition Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
