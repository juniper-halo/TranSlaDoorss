import sys
import traceback
from pathlib import Path

from PIL import Image
from rest_framework import serializers, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

# Ensure project root is importable so we can reach ml_dev
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml_dev.inference.load_best_checkpoint import \
    get_best_checkpoint_path  # noqa: E402
from ml_dev.inference.service import predict_from_file  # noqa: E402

from .models import TrainingFeedback


class PredictionRequestSerializer(serializers.Serializer):
    top_k = serializers.IntegerField(
        required=False, min_value=1, max_value=26, default=1
    )


class TranslatorView(APIView):
    """
    accepts an uploaded image and returns the predicted ASL letter plus confidence.
    """

    def post(self, request: Request):
        image = request.FILES.get("image")
        if image is None:
            return Response(
                {"error": "No submitted image found."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate optional params
        serializer = PredictionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        top_k = serializer.validated_data.get("top_k", 1)

        if not self.verify_image(image):
            return Response(
                {"error": "Invalid/corrupted image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            try:
                best_checkpoint = get_best_checkpoint_path(
                    PROJECT_ROOT / "ml_dev" / "saved_weights"
                )
            except Exception as e:
                print(
                    f"Could not load best checkpoint path: {e}. Falling back to default model."
                )
                best_checkpoint = None
            prediction = predict_from_file(image, top_k=top_k, model_id=best_checkpoint)
        except Exception as exc:
            print("failed to run prediction: ", exc)
            return Response(
                {"error": f"Failed to run prediction: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response({"prediction": prediction}, status=status.HTTP_200_OK)

    def verify_image(self, image):
        try:
            with Image.open(image) as i:
                i.verify()
            image.seek(0)
            return True
        except (OSError, ValueError):
            return False


class FeedbackSerializer(serializers.Serializer):
    predicted_label = serializers.CharField(max_length=1, required=True)
    correct_label = serializers.CharField(max_length=1, required=True)

    def validate_predicted_label(self, value):
        if not value.isupper() or not value.isalpha():
            raise serializers.ValidationError(
                "Predicted label must be a single uppercase letter (A-Z)"
            )
        return value

    def validate_correct_label(self, value):
        if not value.isupper() or not value.isalpha():
            raise serializers.ValidationError(
                "Correct label must be a single uppercase letter (A-Z)"
            )
        return value


class FeedbackView(APIView):
    """
    Accepts user feedback on predictions to improve model training.
    """

    def post(self, request: Request):
        image = request.FILES.get("image")
        if image is None:
            return Response(
                {"error": "No image found."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate feedback data
        serializer = FeedbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        predicted_label = serializer.validated_data["predicted_label"]
        correct_label = serializer.validated_data["correct_label"]

        # Verify image
        if not self.verify_image(image):
            return Response(
                {"error": "Invalid/corrupted image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Save feedback to database
        try:
            feedback = TrainingFeedback(
                image=image,
                predicted_label=predicted_label.upper(),
                correct_label=correct_label.upper(),
            )
            feedback.save()

            return Response(
                {"message": "Feedback received. Thank you!"},
                status=status.HTTP_201_CREATED,
            )
        except Exception as exc:
            # Print full traceback to console
            print("=" * 80)
            print("ERROR SAVING FEEDBACK:")
            print(traceback.format_exc())
            print("=" * 80)
            return Response(
                {"error": f"Failed to save feedback: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def verify_image(self, image):
        try:
            with Image.open(image) as i:
                i.verify()
            image.seek(0)
            return True
        except (OSError, ValueError):
            return False
