"""
lightweight base interface for asl predictors so training and inference code can share a contract
aslpredictor and the evaluator implement this interface today
"""
from __future__ import annotations

from typing import List, Tuple

from PIL import Image


class BaseASLModel:
    """minimal interface for asl predictors"""

    letters: List[str] = [chr(65 + i) for i in range(26)]

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        raise NotImplementedError

    def get_top_predictions(self, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        raise NotImplementedError
