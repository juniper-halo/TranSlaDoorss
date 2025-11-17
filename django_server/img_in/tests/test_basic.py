from hashlib import sha256
import os
import sys
from io import BytesIO

from rest_framework.test import APIClient
from PIL import Image

# Add the root directory to path to import from development
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from ml_dev.development.preprocessing import ASLPreprocessor  # pylint: disable=wrong-import-position
test_path = os.path.dirname(__file__) + "/test_cases/"


def test_translate_valid_image_png():
    # Create new client to send some requests
    client = APIClient()
    # Read the original bytes once, build expected hash from the preprocessed image
    with open(test_path + "png_test.PNG", "rb") as f:
        raw_bytes = f.read()

    # Compute expected hash by preprocessing the image exactly like the server does
    preprocessor = ASLPreprocessor()
    pil_image = Image.open(BytesIO(raw_bytes))
    preprocessed = preprocessor.preprocess(pil_image)
    buf = BytesIO()
    preprocessed.save(buf, format="PNG")
    expected_hash = sha256(buf.getvalue()).hexdigest()

    # POST the original uploaded bytes
    upload_file = BytesIO(raw_bytes)
    upload_file.name = "png_test.PNG"
    response = client.post(
        "/img_in/translate/", {"image": upload_file, "language": "ASL"}, format="multipart"
    )
    try:
        assert response.status_code == 200
    except AssertionError as exc:
        print(response.status_code)
        if "error" in response.data:
            print(response.data["error"])
        raise AssertionError from exc
    assert "translation" in response.data
    assert response.data["translation"] == expected_hash


def test_translate_valid_image_jpg():
    # Create new client to send some requests
    client = APIClient()
    # Read original bytes and compute expected hash after preprocessing
    with open(test_path + "jpg_test.jpg", "rb") as f:
        raw_bytes = f.read()

    preprocessor = ASLPreprocessor()
    pil_image = Image.open(BytesIO(raw_bytes))
    preprocessed = preprocessor.preprocess(pil_image)
    buf = BytesIO()
    preprocessed.save(buf, format="PNG")
    expected_hash = sha256(buf.getvalue()).hexdigest()

    upload_file = BytesIO(raw_bytes)
    upload_file.name = "jpg_test.jpg"
    response = client.post(
        "/img_in/translate/", {"image": upload_file, "language": "ASL"}, format="multipart"
    )
    assert response.status_code == 200
    assert "translation" in response.data
    assert response.data["translation"] == expected_hash


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
