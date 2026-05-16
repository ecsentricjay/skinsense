"""
predict.py
----------
Provides a single public function, predict(), that accepts an image path
(or a PIL Image object) and returns structured inference results from the
SkinSense model.

Usage
-----
    from predict import predict

    result = predict("photo.jpg")
    print(result["top_class"])      # e.g. "Eczema"
    print(result["confidence"])     # e.g. 34.7  (percentage)
    print(result["top3"])           # list of {class, confidence} dicts
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Union

import os

# Suppress verbose TensorFlow startup messages (works on Windows too).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
import tensorflow as tf

from model import CLASS_NAMES, load_model

# ── Image pre-processing constants ─────────────────────────────────────────────

TARGET_SIZE = (224, 224)   # MobileNetV2 expects 224 × 224 inputs.
TOP_K       = 3            # Number of predictions to return.


# ── Model singleton ────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_model() -> tf.keras.Model:
    """Load the model exactly once per process and cache it."""
    return load_model()


# ── Pre-processing ─────────────────────────────────────────────────────────────

def _preprocess(image: Image.Image) -> np.ndarray:
    """
    Resize and normalise a PIL image for MobileNetV2 inference.

    Parameters
    ----------
    image : PIL.Image.Image
        Raw input image in any mode.

    Returns
    -------
    np.ndarray
        Shape (1, 224, 224, 3), dtype float32, values in [0, 1].
    """
    # Ensure RGB — handles RGBA, greyscale, palette images, etc.
    image = image.convert("RGB")
    image = image.resize(TARGET_SIZE, Image.BILINEAR)

    arr = np.array(image, dtype=np.float32) / 255.0   # Normalise to [0, 1].
    return np.expand_dims(arr, axis=0)                 # Add batch dimension.


# ── Public API ─────────────────────────────────────────────────────────────────

def predict(source: Union[str, Path, Image.Image]) -> dict:
    """
    Run inference on a single skin image.

    Parameters
    ----------
    source : str | Path | PIL.Image.Image
        File path to a JPG/PNG image, or an already-opened PIL Image object.

    Returns
    -------
    dict with keys:
        top_class  : str   — Name of the most likely disease class.
        confidence : float — Confidence for the top class (0–100 %).
        top3       : list  — Up to TOP_K dicts, each with keys
                             ``class`` (str) and ``confidence`` (float).
    """
    # Accept a file path or a PIL image directly.
    if isinstance(source, (str, Path)):
        image = Image.open(str(source))
    elif isinstance(source, Image.Image):
        image = source
    else:
        raise TypeError(
            f"source must be a file path or PIL.Image, got {type(source)}"
        )

    # Pre-process and run inference.
    tensor      = _preprocess(image)
    model       = _get_model()
    predictions = model.predict(tensor, verbose=0)[0]   # Shape: (NUM_CLASSES,)

    # Build the ranked top-K result list.
    top_indices = np.argsort(predictions)[::-1][:TOP_K]
    top3 = [
        {
            "class":      CLASS_NAMES[i],
            "confidence": round(float(predictions[i]) * 100, 2),
        }
        for i in top_indices
    ]

    return {
        "top_class":  top3[0]["class"],
        "confidence": top3[0]["confidence"],
        "top3":       top3,
    }
