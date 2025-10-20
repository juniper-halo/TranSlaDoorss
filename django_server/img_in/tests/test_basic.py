import pytest
from rest_framework.test import APIClient
from rest_framework.response import Response
from hashlib import sha256
import os

test_path = os.path.dirname(__file__) + "/test_cases/"


def test_translate_valid_image_png():
    # Create new client to send some requests
    client = APIClient()
    hasher = sha256()
    with open(test_path + "png_test.png", "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
        # With key "image", add png_test.png as raw bytes and also an irrelevant "language" tag
        # POST request to /img_in/translate/
        f.seek(0)
        response = client.post(
            "/img_in/translate/", {"image": f, "language": "ASL"}, format="multipart"
        )
    try:
        assert response.status_code == 200
    except AssertionError:
        print(response.status_code)
        if "error" in response.data:
            print(response.data["error"])
        raise AssertionError
    assert "translation" in response.data
    assert response.data["translation"] == hasher.hexdigest()


def test_translate_valid_image_jpg():
    # Create new client to send some requests
    client = APIClient()
    hasher = sha256()
    with open(test_path + "jpg_test.jpg", "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
        # With key "image", add jpg_test.jpg as raw bytes and also an irrelevant "language" tag
        # POST request to /img_in/translate/
        f.seek(0)
        response = client.post(
            "/img_in/translate/", {"image": f, "language": "ASL"}, format="multipart"
        )
    assert response.status_code == 200
    assert "translation" in response.data

    assert response.data["translation"] == hasher.hexdigest()


def test_translate_invalid_txt():
    # Create new client to send some requests
    client = APIClient()
    with open(test_path + "txt_test.txt", "rb") as f:
        # With key "image", add txt_test.txt as raw bytes and also an irrelevant "language" tag
        # POST request to /img_in/translate/
        response = client.post(
            "/img_in/translate/", {"image": f, "language": "ASL"}, format="multipart"
        )
    assert response.status_code == 400


def test_translate_corrupted_png():
    client = APIClient()
    with open(test_path + "png_test_corrupted.PNG", "rb") as f:
        response = client.post(
            "/img_in/translate/", {"image": f, "language": "ASL"}, format="multipart"
        )
    assert response.status_code == 400


def test_translate_no_image():
    client = APIClient()
    response = client.post(
        "/img_in/translate/", {"lmao": "hehe", "img": "none, :("}, format="multipart"
    )
    assert response.status_code == 400
