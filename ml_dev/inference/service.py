"""
shared inference service helpers for django and cli use

on the ml side, this acts as the python level api surface: it hides model loading,
lets you pass file objects, and returns prediction dicts. django calls this to serve
http requests, and you can call it directly from scripts or notebooks.

how to load a fine-tuned checkpoint in Django (backend playbook):
- set env vars before starting the server:
    export ASL_MODEL_ID="saved_weights/epoch_2"    # path to chosen checkpoint
    export ASL_MODEL_DEVICE="cuda"                 # optional but defaults auto GPU/CPU
- start the server as usual: python manage.py runserver
- warm the cache on startup (recommended) by calling get_predictor() once in your AppConfig.ready():
    from ml_dev.inference.service import get_predictor
    get_predictor()  # uses env vars above; lru_cache keeps it for the process
- each worker process will load once and reuse the model for all requests => restart to change ASL_MODEL_ID.

run the ad-hoc test main (from repo root):
- set ASL_MODEL_ID or pass --model-id
- python -m ml_dev.inference.service --model-id ml_dev/saved_weights/epoch_7 --image /path/to/image.jpg --top-k 3

"""


# from __future__ import annotations

from pathlib import Path
import sys
import io
import os
from functools import lru_cache
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from ml_dev.inference.clip_asl_inference import ASLPredictor


DEFAULT_MODEL_ID = os.environ.get("ASL_MODEL_ID", "openai/clip-vit-base-patch32")
DEFAULT_DEVICE = os.environ.get("ASL_MODEL_DEVICE")


@lru_cache(maxsize=4)
def get_predictor(model_id: str | None = None, device: str | None = None) -> ASLPredictor:
    """
    lazily construct and cache a predictor instance so that model loading
    does not run inside each request path

    args:
        model_id: hugging face repo id or local checkpoint directory defaults to env asl_model_id
        device: torch device override defaults to env asl_model_device or cuda if available
    """
    resolved_model = model_id or DEFAULT_MODEL_ID
    resolved_device = device or DEFAULT_DEVICE
    return ASLPredictor(model_name=resolved_model, device=resolved_device)


def predict_from_file(
    file_obj,
    top_k: int = 1,
    model_id: str | None = None,
    device: str | None = None,
) -> Dict[str, object]:
    """
    run prediction from a django inmemoryuploadedfile or similar file object
    """
    raw_bytes = file_obj.read()
    file_obj.seek(0)
    image = Image.open(io.BytesIO(raw_bytes))

    predictor = get_predictor(model_id=model_id, device=device)

    if top_k == 1:
        letter, confidence = predictor.predict(image)
        return {"letter": letter, "confidence": confidence}

    top_predictions = predictor.get_top_predictions(image, top_k=top_k)
    return {
        "top_predictions": [
            {"letter": letter, "confidence": confidence} for letter, confidence in top_predictions
        ]
    }


# if __name__ == "__main__":

#     # quick manual test harness => remove when integrating into service
#     import argparse
#     parser = argparse.ArgumentParser(description="quick test for ASL predictor")
#     parser.add_argument("--image", required=True, help="path to image file")
#     parser.add_argument("--model-id", default=os.environ.get("ASL_MODEL_ID"), help="checkpoint or hf id")
#     parser.add_argument("--top-k", type=int, default=3, help="top-k predictions to print")
#     args = parser.parse_args()

#     predictor = get_predictor(model_id=args.model_id)
#     with open(args.image, "rb") as fh:
#         if args.top_k == 1:
#             result = predict_from_file(fh, model_id=args.model_id, device=DEFAULT_DEVICE)
#             print(result)
#         else:
#             result = predict_from_file(fh, top_k=args.top_k, model_id=args.model_id, device=DEFAULT_DEVICE)
#             print(result)
