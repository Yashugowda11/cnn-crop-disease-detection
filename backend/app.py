from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import json
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torchvision import models, transforms
except ImportError:
    torch = None
    models = None
    transforms = None

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "plant_disease_resnet18.pth"
METADATA_PATH = MODEL_DIR / "metadata.json"
model = None
model_classes = []
model_class_counts = {}
preprocess = None


def load_model():
    global model, model_classes, model_class_counts, preprocess
    if torch is None or not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return False

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model_classes = metadata["classes"]
    model_class_counts = metadata.get("class_counts", {})
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(model_classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((metadata.get("image_size", 224), metadata.get("image_size", 224))) ,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return True


MODEL_READY = load_model()


def compute_severity_from_image(image_bytes):
    try:
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        leaf_mask = cv2.inRange(hsv, (20, 25, 20), (100, 255, 255)) > 0
        lesion_mask = (
            ((hsv[:, :, 0] < 25) | (hsv[:, :, 0] > 160)) & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] < 220)
        ) | ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 45) & (hsv[:, :, 1] > 55) & (hsv[:, :, 2] < 210))
        leaf_pixels = int(leaf_mask.sum())
        affected_pixels = int((lesion_mask & leaf_mask).sum())
        affected_ratio = affected_pixels / max(leaf_pixels, 1)

        if affected_ratio < 0.08:
            severity = "mild"
        elif affected_ratio < 0.22:
            severity = "moderate"
        else:
            severity = "severe"
        return severity, round(affected_ratio * 100, 2)
    except Exception:
        return "unknown", None


def predict_with_model(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(probabilities, min(5, len(model_classes)))
    top_predictions = [
        {"class": model_classes[index.item()], "probability": round(value.item(), 4)}
        for value, index in zip(values, indices)
    ]
    return top_predictions


def dataset_distribution():
    total = sum(model_class_counts.values())
    if not total:
        return []
    return [
        {
            "class": class_name,
            "count": count,
            "percentage": round(count / total * 100, 2),
        }
        for class_name, count in sorted(model_class_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    ]


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Crop detection backend is running",
        "model_ready": MODEL_READY,
        "classes": len(model_classes),
    })


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    if not MODEL_READY:
        return jsonify({
            "error": "The CNN model is not trained yet. Run backend/train_model.py first.",
            "model_ready": False,
        }), 503

    try:
        top_predictions = predict_with_model(image_bytes)
        severity, affected_area = compute_severity_from_image(image_bytes)
    except Exception as error:
        return jsonify({"error": f"Model inference failed: {error}"}), 500

    prediction = top_predictions[0]
    return jsonify({
        "predicted_class": prediction["class"],
        "confidence": prediction["probability"],
        "top_predictions": top_predictions,
        "dataset_distribution": dataset_distribution(),
        "severity": severity,
        "affected_area_percent": affected_area,
        "recommendation": "Inspect the affected area and consult an agronomist before applying treatment.",
        "explainability": {
            "method": "PlantVillage-trained ResNet18 CNN with HSV lesion-area estimation",
            "heatmap_note": "CNN class probabilities are available; Grad-CAM can be added after model validation.",
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)