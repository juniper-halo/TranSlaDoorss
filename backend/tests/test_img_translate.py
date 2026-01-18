from hashlib import sha256
from io import BytesIO
import sys
from pathlib import Path
from typing import Any, Dict, cast

<<<<<<< HEAD:backend/tests/test_img_translate.py
from django.core.files.uploadedfile import SimpleUploadedFile
=======
import pytest
>>>>>>> 0e240047ddb7361def182b40a7754b344084e6c7:django_server/img_in/tests/test_basic.py
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
<<<<<<< HEAD:backend/tests/test_img_translate.py
    # Read the original bytes once, build expected hash from the preprocessed image
    with open(TEST_PATH / "png_test.PNG", "rb") as f:
=======
    # Read the original bytes once
    with open(test_path + "png_test.PNG", "rb") as f:
>>>>>>> 0e240047ddb7361def182b40a7754b344084e6c7:django_server/img_in/tests/test_basic.py
        raw_bytes = f.read()

    # POST the original uploaded bytes
    upload_file = SimpleUploadedFile("png_test.PNG", raw_bytes, content_type="image/png")
    response = client.post(
        "/img_in/translate/",
        {"image": upload_file, "language": "ASL"},
        format="multipart",
    )
<<<<<<< HEAD:backend/tests/test_img_translate.py
    resp_data = cast(Dict[str, Any], response.data)
    assert response.status_code == 200, resp_data.get("error", response.status_code)
    assert resp_data.get("translation") == expected_hash
=======
    try:
        assert response.status_code == 200
    except AssertionError as exc:
        print(f"Status code: {response.status_code}")
        print(f"Response data: {response.data}")
        print(f"Response content: {response.content}")
        raise AssertionError from exc
    assert "prediction" in response.data
    prediction = response.data["prediction"]
    assert "letter" in prediction
    assert "confidence" in prediction
    assert isinstance(prediction["letter"], str)
    assert len(prediction["letter"]) == 1
    assert prediction["letter"].isupper()
    assert isinstance(prediction["confidence"], (float, int))
    assert 0 <= prediction["confidence"] <= 1
>>>>>>> 0e240047ddb7361def182b40a7754b344084e6c7:django_server/img_in/tests/test_basic.py


def test_translate_valid_image_jpg():
    # Create new client to send some requests
    client = APIClient()
<<<<<<< HEAD:backend/tests/test_img_translate.py
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
=======
    # Read original bytes
    with open(test_path + "jpg_test.jpg", "rb") as f:
        raw_bytes = f.read()

    upload_file = BytesIO(raw_bytes)
    upload_file.name = "jpg_test.jpg"
>>>>>>> 0e240047ddb7361def182b40a7754b344084e6c7:django_server/img_in/tests/test_basic.py
    response = client.post(
        "/img_in/translate/",
        {"image": upload_file, "language": "ASL"},
        format="multipart",
    )
<<<<<<< HEAD:backend/tests/test_img_translate.py
    resp_data = cast(Dict[str, Any], response.data)
    assert response.status_code == 200, resp_data.get("error", response.status_code)
    assert resp_data.get("translation") == expected_hash
=======
    if response.status_code != 200:
        print(f"Status code: {response.status_code}")
        print(f"Response data: {response.data}")
        print(f"Response content: {response.content}")
    assert response.status_code == 200
    assert "prediction" in response.data
    prediction = response.data["prediction"]
    assert "letter" in prediction
    assert "confidence" in prediction
    assert isinstance(prediction["letter"], str)
    assert len(prediction["letter"]) == 1
    assert prediction["letter"].isupper()
    assert isinstance(prediction["confidence"], (float, int))
    assert 0 <= prediction["confidence"] <= 1
>>>>>>> 0e240047ddb7361def182b40a7754b344084e6c7:django_server/img_in/tests/test_basic.py


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


@pytest.mark.django_db
def test_feedback_valid_correct():
    """Test submitting feedback when prediction was correct"""
    client = APIClient()
    with open(test_path + "png_test.PNG", "rb") as f:
        raw_bytes = f.read()

    upload_file = BytesIO(raw_bytes)
    upload_file.name = "feedback_test.png"
    response = client.post(
        "/img_in/feedback/",
        {
            "image": upload_file,
            "predicted_label": "A",
            "correct_label": "A",
        },
        format="multipart",
    )
    assert response.status_code == 201
    assert "message" in response.data


@pytest.mark.django_db
def test_feedback_valid_correction():
    """Test submitting feedback with a correction"""
    client = APIClient()
    with open(test_path + "jpg_test.jpg", "rb") as f:
        raw_bytes = f.read()

    upload_file = BytesIO(raw_bytes)
    upload_file.name = "feedback_correction.jpg"
    response = client.post(
        "/img_in/feedback/",
        {
            "image": upload_file,
            "predicted_label": "B",
            "correct_label": "C",
        },
        format="multipart",
    )
    assert response.status_code == 201
    assert "message" in response.data


@pytest.mark.django_db
def test_feedback_no_image():
    """Test feedback submission fails without image"""
    client = APIClient()
    response = client.post(
        "/img_in/feedback/",
        {
            "predicted_label": "D",
            "correct_label": "E",
        },
        format="multipart",
    )
    assert response.status_code == 400
    assert "error" in response.data


@pytest.mark.django_db
def test_feedback_invalid_image():
    """Test feedback submission fails with invalid image"""
    client = APIClient()
    with open(test_path + "txt_test.txt", "rb") as f:
        response = client.post(
            "/img_in/feedback/",
            {
                "image": f,
                "predicted_label": "F",
                "correct_label": "G",
            },
            format="multipart",
        )
    assert response.status_code == 400
    assert "error" in response.data


@pytest.mark.django_db
def test_feedback_missing_labels():
    """Test feedback submission fails without required labels"""
    client = APIClient()
    with open(test_path + "png_test.PNG", "rb") as f:
        raw_bytes = f.read()

    upload_file = BytesIO(raw_bytes)
    upload_file.name = "test.png"

    # Missing correct_label
    response = client.post(
        "/img_in/feedback/",
        {
            "image": upload_file,
            "predicted_label": "H",
        },
        format="multipart",
    )
    assert response.status_code == 400


@pytest.mark.django_db
def test_feedback_invalid_label_format():
    """Test feedback submission validates label format"""
    client = APIClient()
    with open(test_path + "png_test.PNG", "rb") as f:
        raw_bytes = f.read()

    upload_file = BytesIO(raw_bytes)
    upload_file.name = "test.png"

    # Lowercase letter should fail
    response = client.post(
        "/img_in/feedback/",
        {
            "image": upload_file,
            "predicted_label": "a",
            "correct_label": "B",
        },
        format="multipart",
    )
    assert response.status_code == 400
