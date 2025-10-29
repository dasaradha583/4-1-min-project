#!/usr/bin/env python3
"""
api.py — FastAPI backend for wheat disease detection
Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
Then test:
    curl -X POST -F "file=@Split Dataset/Test/LeafBlight/LB_001.jpg" http://127.0.0.1:8000/predict
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image
import numpy as np
import json
from pathlib import Path
from .train import ToTensorNormalize, Resize, SmallCNN, pil_load_rgb

app = FastAPI(title="🌾 Wheat Disease Classifier")

# --- Load model once ---
MODEL_DIR = Path("experiments")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    ts = MODEL_DIR / "model_scripted.pt"
    best = MODEL_DIR / "best_model.pth"
    meta = MODEL_DIR / "metadata.json"
    classes = None
    if meta.exists():
        classes = json.load(open(meta))["classes"]

    if ts.exists():
        model = torch.jit.load(str(ts), map_location=DEVICE)
        model.eval()
        print("Loaded TorchScript model.")
        return model, classes
    elif best.exists():
        ckpt = torch.load(str(best), map_location=DEVICE)
        model = SmallCNN(num_classes=len(classes))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE).eval()
        print("Loaded best_model.pth.")
        return model, classes
    else:
        raise FileNotFoundError("Model not found in experiments/")

model, CLASS_NAMES = load_model()

# --- Inference utils ---
def preprocess(image: Image.Image):
    t = ToTensorNormalize()
    r = Resize((224, 224))
    return t(r(image)).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = preprocess(image).to(DEVICE)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1)
            pred_idx = int(probs.argmax(dim=1).item())
            confidence = float(probs.max().item())
        
        # Confidence threshold for out-of-distribution detection
        CONFIDENCE_THRESHOLD = 0.65  # 65% threshold
        
        if confidence < CONFIDENCE_THRESHOLD:
            result = {
                "filename": file.filename,
                "prediction": "Unknown",
                "confidence": round(confidence * 100, 2),
                "message": "Low confidence - image may not be wheat disease. Please upload a clear wheat image."
            }
        else:
            result = {
                "filename": file.filename,
                "prediction": CLASS_NAMES[pred_idx],
                "confidence": round(confidence * 100, 2),
            }
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "Wheat Disease Classifier API is running!"}
