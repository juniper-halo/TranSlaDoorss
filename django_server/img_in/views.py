from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpRequest
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework import status
from PIL import Image
from hashlib import sha256


# Create your views here.
class TranslatorView(APIView):
    def post(self, request: Request):
        image = request.FILES.get("image")
        if image is None:
            return Response(
                {"error": "No submitted image found."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        elif self.verify_image(image):
            pil_image = Image.open(image)
        else:
            return Response(
                {"error": "Invalid/corrupted image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        hash_thingy = sha256()
        for chunk in image.chunks():
            hash_thingy.update(chunk)
        final_hash = hash_thingy.hexdigest()

        return Response({"translation": final_hash})

    def verify_image(self, image):
        try:
            i = Image.open(image)
            i.verify()
            image.seek(0)
            return True
        except Exception:
            return False
