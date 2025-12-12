import sys
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

from ml_dev.inference.service import predict_from_file  # noqa: E402


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
            prediction = predict_from_file(image, top_k=top_k)
        except Exception as exc:
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
