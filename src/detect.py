#!/usr/bin/env python3
"""
detect.py
Run disease prediction on one image or an entire folder.
Uses model_scripted.pt if available, otherwise best_model.pth.
Example:
  python src/detect.py --image "Split Dataset/Test/LeafBlight/LB_001.jpg"
  python src/detect.py --image "Split Dataset/Test/LeafBlight"
"""

import argparse
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import json
from train import ToTensorNormalize, Resize, pil_load_rgb, SmallCNN  # reuse from training

def load_model(model_dir, device):
    """Load TorchScript model if available, else load best_model.pth"""
    model_dir = Path(model_dir)
    ts = model_dir / "model_scripted.pt"
    best = model_dir / "best_model.pth"
    meta = model_dir / "metadata.json"

    classes = None
    if meta.exists():
        with open(meta, 'r') as f:
            classes = json.load(f).get("classes")

    if ts.exists():
        model = torch.jit.load(str(ts), map_location=device)
        model.eval()
        print("Loaded TorchScript model.")
        return model, classes
    elif best.exists():
        ckpt = torch.load(str(best), map_location=device)
        classes = ckpt.get("classes", classes)
        model = SmallCNN(num_classes=len(classes))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        print("Loaded best_model.pth successfully.")
        return model, classes
    else:
        raise FileNotFoundError("No model found in 'experiments' (expected model_scripted.pt or best_model.pth).")

def preprocess(img_path):
    img = pil_load_rgb(img_path)
    transform = lambda img: ToTensorNormalize()(Resize((224, 224))(img))
    return transform(img).unsqueeze(0)

def predict(model, tensor, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs.max().item()
    return pred_idx, confidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image or directory of images")
    parser.add_argument("--model_dir", default="experiments", help="Path to folder with trained model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, classes = load_model(args.model_dir, device)
    if classes is None:
        raise RuntimeError("Class labels missing in metadata.json")

    inp = Path(args.image)
    if not inp.exists():
        raise FileNotFoundError(f"Image or folder not found: {inp}")

    # Collect image paths
    image_paths = []
    if inp.is_file():
        image_paths = [inp]
    else:
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            image_paths.extend(sorted(inp.glob(f"*{ext}")))

    if not image_paths:
        print("No images found.")
        return

    for img_path in image_paths:
        tensor = preprocess(img_path)
        idx, conf = predict(model, tensor, device)
        label = classes[idx] if idx < len(classes) else "Unknown"
        print(f"{img_path.name} → {label}  (confidence: {conf*100:.2f}%)")

if __name__ == "__main__":
    main()
