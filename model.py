"""
model.py
--------
Defines and initialises the SkinSense transfer-learning model.

Architecture:
    MobileNetV2 (frozen ImageNet weights)
    → GlobalAveragePooling2D
    → Dense(128, relu)
    → Dropout(0.3)
    → Dense(8, softmax)

Run this file directly to (re-)generate model_architecture.json and
model_weights.h5 in the current directory.

NOTE: The weights saved here are random initialisation weights suitable
      for a demonstration prototype.  Replace model_weights.h5 with
      weights from a properly labelled dermatology dataset before any
      clinical or research use.
"""

import json
import os

# Suppress verbose TensorFlow startup messages (works on Windows too).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ── Constants ──────────────────────────────────────────────────────────────────

INPUT_SHAPE = (224, 224, 3)

CLASS_NAMES = [
    "Eczema",
    "Tinea Versicolor",
    "Acne Vulgaris",
    "Vitiligo",
    "Psoriasis",
    "Melanoma",
    "Acne Keloidalis Nuchae",
    "Seborrheic Dermatitis",
]

NUM_CLASSES = len(CLASS_NAMES)

ARCH_PATH    = "model_architecture.json"
WEIGHTS_PATH = "model_weights.h5"


# ── Model builder ──────────────────────────────────────────────────────────────

def build_model() -> tf.keras.Model:
    """Return a compiled Keras model with MobileNetV2 as the frozen backbone."""

    # Load MobileNetV2 without the top classification head.
    base = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Freeze all base layers for demo purposes.

    # Build the classification head.
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="image_input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SkinSense")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ── Persistence helpers ────────────────────────────────────────────────────────

def save_model(model: tf.keras.Model) -> None:
    """Serialise architecture to JSON and weights to HDF5."""
    with open(ARCH_PATH, "w") as fh:
        json.dump(json.loads(model.to_json()), fh, indent=2)
    model.save_weights(WEIGHTS_PATH)
    print(f"[model.py] Architecture → {ARCH_PATH}")
    print(f"[model.py] Weights      → {WEIGHTS_PATH}")


def load_model() -> tf.keras.Model:
    """
    Rebuild the model from the saved architecture + weights files.

    Returns
    -------
    tf.keras.Model
        Model with weights loaded and ready for inference.
    """
    if not os.path.exists(ARCH_PATH):
        raise FileNotFoundError(
            f"Architecture file not found: {ARCH_PATH}\n"
            "Run `python model.py` first to generate it."
        )
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Weights file not found: {WEIGHTS_PATH}\n"
            "Run `python model.py` first to generate it."
        )

    with open(ARCH_PATH, "r") as fh:
        model = tf.keras.models.model_from_json(json.dumps(json.load(fh)))

    model.load_weights(WEIGHTS_PATH)
    return model


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[model.py] Building model …")
    model = build_model()
    model.summary()
    save_model(model)
    print("[model.py] Done.")
