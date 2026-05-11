"""
Plant Disease Detection - Inference Utility
"""

import json
import numpy as np
import tensorflow as tf
from PIL import Image
import io

MODEL_PATH       = "model/saved_model/plant_disease_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"
IMG_SIZE         = (224, 224)
CONFIDENCE_THRESHOLD = 0.5


def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, H, W, 3)


def predict(model, class_names: list, image_bytes: bytes) -> dict:
    arr  = preprocess_image(image_bytes)
    preds = model.predict(arr, verbose=0)[0]          # shape: (num_classes,)

    top5_idx = np.argsort(preds)[::-1][:5]
    top5 = [
        {"class": class_names[i], "confidence": float(preds[i])}
        for i in top5_idx
    ]

    predicted_class = class_names[top5_idx[0]]
    confidence      = float(preds[top5_idx[0]])

    # Parse plant & disease from class name like "Tomato___Early_blight"
    parts = predicted_class.replace("___", " | ").replace("_", " ").split(" | ")
    plant   = parts[0] if len(parts) > 0 else "Unknown"
    disease = parts[1] if len(parts) > 1 else "Unknown"

    return {
        "plant":      plant,
        "disease":    disease,
        "confidence": round(confidence * 100, 2),
        "is_healthy": "healthy" in predicted_class.lower(),
        "low_confidence": confidence < CONFIDENCE_THRESHOLD,
        "top5": top5,
    }
