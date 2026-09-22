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

DISEASE_GUIDANCE = {
    "healthy": {
        "solution": "No disease pattern was identified. Continue regular scouting, balanced watering, and good crop hygiene.",
        "pesticide": "No pesticide is recommended for a healthy leaf. Treat only after confirming a disease and its cause.",
    },
    "Apple___Apple_scab": {
        "solution": "Remove fallen leaves and infected fruit, prune for airflow, and avoid overhead irrigation.",
        "pesticide": "Use a locally approved captan or myclobutanil fungicide at the label rate and correct growth stage.",
    },
    "Apple___Black_rot": {
        "solution": "Prune dead or infected branches, remove mummified fruit, and disinfect pruning tools between cuts.",
        "pesticide": "A labeled copper or captan fungicide may help; follow the product label and harvest interval.",
    },
    "Apple___Cedar_apple_rust": {
        "solution": "Remove infected leaves, improve canopy ventilation, and remove nearby juniper hosts where appropriate.",
        "pesticide": "Use an approved myclobutanil or mancozeb fungicide preventively according to the label.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "solution": "Prune crowded growth, improve airflow, and remove severely infected leaves.",
        "pesticide": "A labeled sulfur, potassium bicarbonate, or approved fungicide can be used as directed.",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "solution": "Use resistant seed, rotate crops, manage plant residue, and avoid prolonged leaf wetness.",
        "pesticide": "For serious infections, use a locally approved strobilurin or triazole fungicide according to its label.",
    },
    "Corn_(maize)___Common_rust_": {
        "solution": "Plant resistant varieties, scout surrounding plants, and remove heavily infected debris after harvest.",
        "pesticide": "An approved triazole or strobilurin fungicide may be used when disease pressure is high.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "solution": "Use resistant hybrids, rotate away from corn, bury residue, and improve field airflow.",
        "pesticide": "Use a locally approved azoxystrobin, propiconazole, or equivalent fungicide only as labeled.",
    },
    "Grape___Black_rot": {
        "solution": "Remove mummified berries and infected leaves, prune the canopy, and improve sunlight and airflow.",
        "pesticide": "A labeled myclobutanil, captan, or mancozeb fungicide may be appropriate at the recommended stage.",
    },
    "Grape___Esca_(Black_Measles)": {
        "solution": "Remove and destroy severely affected wood, disinfect pruning tools, and avoid spreading infected material.",
        "pesticide": "There is no reliable curative pesticide; consult a viticulture specialist about sanitation and vine replacement.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "solution": "Remove infected leaves, reduce canopy humidity, improve ventilation, and avoid overhead irrigation.",
        "pesticide": "A labeled copper, mancozeb, or other approved grape fungicide may help prevent spread.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "solution": "Confirm the diagnosis with an agricultural service, remove infected trees where required, and control the psyllid vector.",
        "pesticide": "There is no pesticide cure; use only locally approved psyllid-management products under expert guidance.",
    },
    "Peach___Bacterial_spot": {
        "solution": "Use resistant varieties, improve drainage and airflow, remove severely affected material, and avoid wet foliage.",
        "pesticide": "A labeled copper bactericide may reduce spread when applied preventively and according to local guidance.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "solution": "Use clean seed and transplants, remove infected debris, rotate crops, and avoid handling wet plants.",
        "pesticide": "A labeled copper-based bactericide may help suppress spread; follow resistance-management instructions.",
    },
    "Potato___Early_blight": {
        "solution": "Remove infected foliage, rotate crops, mulch soil, maintain plant nutrition, and water at the base.",
        "pesticide": "Approved chlorothalonil, mancozeb, or copper fungicides may help when used as directed.",
    },
    "Potato___Late_blight": {
        "solution": "Remove infected plants, avoid overhead irrigation, improve airflow, and destroy infected tubers and debris.",
        "pesticide": "Use an approved late-blight fungicide such as chlorothalonil or a locally registered equivalent immediately after confirmation.",
    },
    "Squash___Powdery_mildew": {
        "solution": "Increase spacing and airflow, remove badly infected leaves, and water the root zone consistently.",
        "pesticide": "Labeled sulfur, potassium bicarbonate, or neem products may help when used according to the crop label.",
    },
    "Strawberry___Leaf_scorch": {
        "solution": "Remove infected leaves, improve spacing and airflow, control weeds, and avoid prolonged leaf wetness.",
        "pesticide": "Use only a locally approved strawberry fungicide, such as a labeled captan product, as directed.",
    },
    "Tomato___Bacterial_spot": {
        "solution": "Remove infected leaves, use clean transplants, rotate crops, and avoid working with wet foliage.",
        "pesticide": "A labeled copper bactericide may suppress spread; follow the label and resistance-management guidance.",
    },
    "Tomato___Early_blight": {
        "solution": "Remove lower infected leaves, mulch soil, improve airflow, rotate crops, and water at the plant base.",
        "pesticide": "Labeled chlorothalonil, mancozeb, or copper fungicides may help prevent new infections.",
    },
    "Tomato___Late_blight": {
        "solution": "Remove and destroy infected plants, isolate the area, improve airflow, and avoid wetting leaves.",
        "pesticide": "Use a locally registered late-blight fungicide promptly after confirmation and follow the label strictly.",
    },
    "Tomato___Leaf_Mold": {
        "solution": "Increase ventilation, reduce humidity, space plants properly, and remove infected leaves.",
        "pesticide": "An approved copper or chlorothalonil fungicide may help when used with proper crop and harvest intervals.",
    },
    "Tomato___Septoria_leaf_spot": {
        "solution": "Remove spotted lower leaves, mulch to prevent soil splash, rotate crops, and water at the base.",
        "pesticide": "Labeled chlorothalonil, copper, or mancozeb products may protect new foliage when used as directed.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "solution": "Rinse leaf undersides, reduce plant stress, remove heavily infested leaves, and protect beneficial predators.",
        "pesticide": "Use a crop-approved insecticidal soap or miticide only if needed, following label and re-entry instructions.",
    },
    "Tomato___Target_Spot": {
        "solution": "Remove infected debris, improve spacing and airflow, rotate crops, and avoid overhead watering.",
        "pesticide": "An approved chlorothalonil, mancozeb, or strobilurin fungicide may help prevent spread.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "solution": "Remove severely infected plants, control whiteflies, use insect-proof nursery practices, and plant resistant varieties.",
        "pesticide": "There is no pesticide cure for the virus; use only approved whitefly-control products under local guidance.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "solution": "Remove infected plants, disinfect tools and hands, control weeds, and avoid saving seed from affected plants.",
        "pesticide": "There is no pesticide cure; manage insect vectors if present and use certified disease-free seed.",
    },
}


def guidance_for(class_name):
    if class_name in DISEASE_GUIDANCE:
        return DISEASE_GUIDANCE[class_name]
    if class_name and class_name.endswith("___healthy"):
        return DISEASE_GUIDANCE["healthy"]
    return {
        "solution": "Isolate the affected plant, remove severely damaged leaves, improve airflow, and confirm the diagnosis locally.",
        "pesticide": "Do not apply a product based only on this screen. Ask a local agricultural expert for a crop-approved treatment.",
    }


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
    guidance = guidance_for(prediction["class"])
    return jsonify({
        "predicted_class": prediction["class"],
        "confidence": prediction["probability"],
        "top_predictions": top_predictions,
        "dataset_distribution": dataset_distribution(),
        "severity": severity,
        "affected_area_percent": affected_area,
        "recommendation": guidance["solution"],
        "pesticide": guidance["pesticide"],
        "explainability": {
            "method": "PlantVillage-trained ResNet18 CNN with HSV lesion-area estimation",
            "heatmap_note": "CNN class probabilities are available; Grad-CAM can be added after model validation.",
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)