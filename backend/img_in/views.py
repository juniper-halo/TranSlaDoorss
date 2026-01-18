import sys
from pathlib import Path
from typing import Any, Dict, List, cast

from PIL import Image
from rest_framework import serializers, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

# Ensure project root is importable so we can reach ml
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml.inference.service import predict_from_file  # noqa: E402
from .models import TrainingFeedback


class PredictionRequestSerializer(serializers.Serializer):
    top_k = serializers.IntegerField(required=False, min_value=1, max_value=26, default=1)


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
            prediction: Dict[str, Any] = predict_from_file(image, top_k=top_k)
        except Exception as exc:
            print("failed to run prediction: ", exc)
            return Response(
                {"error": f"Failed to run prediction: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # sync response contract with frontend/tests: always include "translation"
        if top_k == 1:
            letter = prediction.get("letter")
            payload = {
                "translation": letter,
                "confidence": prediction.get("confidence"),
                "prediction": prediction,
            }
        else:
            top_preds: List[Dict[str, Any]] = cast(List[Dict[str, Any]], prediction.get("top_predictions") or [])
            best_letter = top_preds[0].get("letter") if top_preds else None
            payload = {
                "translation": best_letter,
                "top_predictions": top_preds,
                "prediction": prediction,
            }

        return Response(payload, status=status.HTTP_200_OK)

    def verify_image(self, image) -> bool:
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

    def validate_predicted_label(self, value: str) -> str:
        if not value.isalpha() or len(value) != 1:
            raise serializers.ValidationError("Predicted label must be a single letter (A-Z)")
        return value.upper()

    def validate_correct_label(self, value: str) -> str:
        if not value.isalpha() or len(value) != 1:
            raise serializers.ValidationError("Correct label must be a single letter (A-Z)")
        return value.upper()


class FeedbackView(APIView):
    """
    Accepts user feedback on predictions and stores uploaded frames for future retraining.
    """

    def post(self, request: Request):
        image = request.FILES.get("image")
        if image is None:
            return Response({"error": "No image found."}, status=status.HTTP_400_BAD_REQUEST)

        serializer = FeedbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        if not self.verify_image(image):
            return Response({"error": "Invalid/corrupted image."}, status=status.HTTP_400_BAD_REQUEST)

        feedback = TrainingFeedback.objects.create(
            image=image,
            predicted_label=serializer.validated_data["predicted_label"],
            correct_label=serializer.validated_data["correct_label"],
        )
        return Response(
            {
                "message": "Feedback received. Thank you!",
                "id": feedback.id,
                "predicted_label": feedback.predicted_label,
                "correct_label": feedback.correct_label,
            },
            status=status.HTTP_201_CREATED,
        )

    def verify_image(self, image) -> bool:
        try:
            with Image.open(image) as i:
                i.verify()
            image.seek(0)
            return True
        except (OSError, ValueError):
            return False
