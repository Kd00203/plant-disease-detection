"""
Plant Disease Detection — Flask REST API
Run: python api/app.py
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.predict import load_model_and_classes, predict

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Load model once at startup
print("Loading model...")
model, class_names = load_model_and_classes()
print(f"Model loaded. {len(class_names)} classes.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict_route():
    """
    POST /api/predict
    Body: multipart/form-data with field 'image' (file)
    Returns JSON with prediction results
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, or WEBP."}), 400

    image_bytes = file.read()

    try:
        result = predict(model, class_names, image_bytes)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "success":    True,
        "plant":      result["plant"],
        "disease":    result["disease"],
        "confidence": result["confidence"],
        "is_healthy": result["is_healthy"],
        "warning":    "Low confidence prediction" if result["low_confidence"] else None,
        "top5":       result["top5"],
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": len(class_names)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
