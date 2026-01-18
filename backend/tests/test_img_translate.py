from hashlib import sha256
from io import BytesIO
import sys
from pathlib import Path
from typing import Any, Dict, cast

from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
from rest_framework.test import APIClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.development.preprocessing import ASLPreprocessor  # pylint: disable=wrong-import-position

TEST_PATH = Path(__file__).parent / "test_cases"


def test_translate_valid_image_png():
    # Create new client to send some requests
    client = APIClient()
    # Read the original bytes once, build expected hash from the preprocessed image
    with open(TEST_PATH / "png_test.PNG", "rb") as f:
        raw_bytes = f.read()

    # Compute expected hash by preprocessing the image exactly like the server does
    preprocessor = ASLPreprocessor()
    pil_image = Image.open(BytesIO(raw_bytes))
    preprocessed = preprocessor.preprocess(pil_image)
    buf = BytesIO()
    preprocessed.save(buf, format="PNG")
    expected_hash = sha256(buf.getvalue()).hexdigest()

    # POST the original uploaded bytes
    upload_file = SimpleUploadedFile("png_test.PNG", raw_bytes, content_type="image/png")
    response = client.post(
        "/img_in/translate/",
        {"image": upload_file, "language": "ASL"},
        format="multipart",
    )
    resp_data = cast(Dict[str, Any], response.data)
    assert response.status_code == 200, resp_data.get("error", response.status_code)
    assert resp_data.get("translation") == expected_hash


def test_translate_valid_image_jpg():
    # Create new client to send some requests
    client = APIClient()
    # Read original bytes and compute expected hash after preprocessing
    with open(TEST_PATH / "jpg_test.jpg", "rb") as f:
        raw_bytes = f.read()

    preprocessor = ASLPreprocessor()
    pil_image = Image.open(BytesIO(raw_bytes))
    preprocessed = preprocessor.preprocess(pil_image)
    buf = BytesIO()
    preprocessed.save(buf, format="PNG")
    expected_hash = sha256(buf.getvalue()).hexdigest()

    upload_file = SimpleUploadedFile("jpg_test.jpg", raw_bytes, content_type="image/jpeg")
    response = client.post(
        "/img_in/translate/",
        {"image": upload_file, "language": "ASL"},
        format="multipart",
    )
    resp_data = cast(Dict[str, Any], response.data)
    assert response.status_code == 200, resp_data.get("error", response.status_code)
    assert resp_data.get("translation") == expected_hash


def test_translate_invalid_txt():
    # Create new client to send some requests
    client = APIClient()
    with open(TEST_PATH / "txt_test.txt", "rb") as f:
        # With key "image", add txt_test.txt as raw bytes and also an irrelevant "language" tag
        # POST request to /img_in/translate/
        response = client.post(
            "/img_in/translate/", {"image": f, "language": "ASL"}, format="multipart"
        )
    assert response.status_code == 400


def test_translate_corrupted_png():
    client = APIClient()
    with open(TEST_PATH / "png_test_corrupted.PNG", "rb") as f:
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
