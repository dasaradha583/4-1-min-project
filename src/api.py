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
import cv2
from pathlib import Path
from .train import ToTensorNormalize, Resize, SmallCNN, pil_load_rgb

def validate_wheat_image(image_array):
    """Comprehensive validation for wheat plant images."""
    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # 1. Color Distribution Check
        # Define broader plant color ranges in HSV
        green_lower = np.array([25, 20, 20])
        green_upper = np.array([95, 255, 255])
        yellow_lower = np.array([15, 30, 30])
        yellow_upper = np.array([35, 255, 255])
        brown_lower = np.array([10, 20, 20])
        brown_upper = np.array([25, 255, 200])

        # Create masks for each color range
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

        # Calculate color percentages
        total_pixels = image_array.shape[0] * image_array.shape[1]
        plant_pixels = np.sum(green_mask > 0) + np.sum(yellow_mask > 0) + np.sum(brown_mask > 0)
        plant_ratio = plant_pixels / total_pixels

        # 2. Texture Analysis
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 3. Edge Detection for plant-like structures
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / total_pixels

        # 4. Check for faces (to reject human/animal photos)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        has_faces = len(faces) > 0

        # 5. Check image properties typical for wheat disease photos
        is_closeup = min(image_array.shape[0], image_array.shape[1]) >= 224
        
        # Combine all checks
        is_valid = (
            plant_ratio > 0.3 and           # At least 30% plant colors
            texture_score > 100 and         # Sufficient texture variation
            edge_density > 0.05 and         # Sufficient edge features
            not has_faces and              # No faces detected
            is_closeup                      # Image is detailed enough
        )

        return is_valid, {
            "has_faces": has_faces,
            "plant_color_ratio": plant_ratio,
            "texture_score": texture_score,
            "edge_density": edge_density,
            "is_closeup": is_closeup
        }
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return False, {}

def check_if_plant_image(image_array):
    """Basic check for plant-like characteristics using color and texture."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    # Define green color range in HSV
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([100, 255, 255])
    
    # Create mask for green colors
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Calculate percentage of green pixels
    green_pixel_ratio = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
    
    # Calculate texture features using grayscale image
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Return True if the image has significant green content and texture variation
    return green_pixel_ratio > 0.15 and texture_score > 100

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
        # Validate file type
        if not file.content_type.startswith('image/'):
            return JSONResponse({
                "error": "Invalid file type. Please upload an image file.",
                "details": "Supported formats: JPG, JPEG, PNG"
            }, status_code=400)

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Basic image validation
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Convert PIL Image to numpy array for validation
        image_array = np.array(image)
        
        # Comprehensive wheat image validation
        is_valid, validation_details = validate_wheat_image(image_array)
        
        if not is_valid:
            error_message = "This image doesn't appear to be a wheat plant photo. Please ensure:"
            details = []
            
            if validation_details.get("has_faces", False):
                details.append("- No human faces or people in the image")
            if validation_details.get("plant_color_ratio", 0) < 0.3:
                details.append("- Image contains predominantly plant colors (green, yellow, or brown)")
            if not validation_details.get("is_closeup", False):
                details.append("- Image is a close-up shot of the plant")
            if validation_details.get("texture_score", 0) < 100:
                details.append("- Clear focus on plant texture and details")
                
            return JSONResponse({
                "error": error_message,
                "details": details,
                "prediction": "Unknown",
                "confidence": 0
            }, status_code=400)
            
        tensor = preprocess(image).to(DEVICE)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1)
            
            # Get all prediction probabilities
            all_probs = probs[0].tolist()
            entropy = -sum(p * torch.log2(torch.tensor(p + 1e-9)) for p in all_probs)
            
            # Get top 3 predictions and their confidences
            top_probs, top_indices = torch.topk(probs, k=3, dim=1)
            confidence1 = float(top_probs[0][0].item())
            confidence2 = float(top_probs[0][1].item())
            confidence3 = float(top_probs[0][2].item())
            
            # Calculate prediction uncertainty metrics
            top_uncertainty = confidence2 / confidence1  # Ratio of second highest to highest confidence
            spread_uncertainty = (confidence1 - confidence3) / confidence1  # Spread of confidences
            
            # Extremely strict validation criteria
            CONFIDENCE_THRESHOLD = 0.85  # 85% threshold
            ENTROPY_THRESHOLD = 1.0  # Lower entropy means more certain prediction
            
            # Check for signs of uncertainty
            is_uncertain = (
                confidence1 < CONFIDENCE_THRESHOLD or  # Main confidence too low
                entropy > ENTROPY_THRESHOLD or  # High entropy in distribution
                top_uncertainty > 0.5 or  # Second prediction too close to first
                spread_uncertainty < 0.4  # Top predictions too similar
            )
            
            if is_uncertain:
                return JSONResponse({
                    "filename": file.filename,
                    "prediction": "Unknown",
                    "confidence": round(confidence1 * 100, 2),
                    "message": "Cannot confidently classify this image. Please ensure you're uploading a clear image of wheat disease symptoms."
                })
            
            result = {
                "filename": file.filename,
                "prediction": CLASS_NAMES[int(top_indices[0][0].item())],
                "confidence": round(confidence1 * 100, 2),
            }
            return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "Wheat Disease Classifier API is running!"}
