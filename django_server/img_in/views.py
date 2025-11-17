import os
import sys
from hashlib import sha256
from io import BytesIO

from PIL import Image
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

# Add the root directory to path to import from development
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ml_dev.development.preprocessing import \
    ASLPreprocessor  # pylint: disable=wrong-import-position


# Create your views here.
class TranslatorView(APIView):
    def post(self, request: Request):
        image = request.FILES.get("image")
        if image is None:
            return Response(
                {"error": "No submitted image found."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if self.verify_image(image):
            # pil_image = Image.open(image)
            pass

        else:
            return Response(
                {"error": "Invalid/corrupted image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Preprocess the image using ASLPreprocessor
        preprocessor = ASLPreprocessor()
        pil_image = Image.open(image)
        preprocessed_image = preprocessor.preprocess(pil_image)

        # Convert preprocessed image to bytes for hashing
        buf = BytesIO()
        preprocessed_image.save(buf, format="PNG")
        data = buf.getvalue()

        # Hash the preprocessed image to verify that it worked
        hash_thingy = sha256()
        hash_thingy.update(data)
        final_hash = hash_thingy.hexdigest()

        return Response({"translation": final_hash})

    def verify_image(self, image):
        try:
            i = Image.open(image)
            i.verify()
            image.seek(0)
            return True
        except (OSError, ValueError):
            return False
