# predictor.py (Revised Content)

import torch
from torchvision import transforms
from PIL import Image
from model_loader import load_model 
from io import BytesIO

# 1. CONFIGURATION (Ensure these are accurate to your model)
DEVICE = torch.device("cpu") 
MODEL_PATH = "model_trained/hybrid_densenet_inception_resse_final.pth" 
CLASS_NAMES = [
    'Class_1_Healthy', 'Class_2_Mild', 'Class_3_Moderate', 'Class_4_Severe', 
    'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9', 
    'Class_10', 'Class_11', 'Class_12', 'Class_13_Deadliest'
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 299

# ... (Model and Transform loading code remains the same) ...

def predict_crop_and_severity(image_bytes: BytesIO):
    """Takes image bytes, returns only the primary prediction and severity score."""
    if MODEL_LOAD_STATUS != "success":
        return {"error": MODEL_LOAD_STATUS, "status": "error"}
        
    try:
        img = Image.open(image_bytes).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {e}", "status": "error"}

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        
        # 1. Primary Prediction
        conf, pred_idx = torch.max(probs, dim=1)
        
        pred_class = CLASS_NAMES[pred_idx.item()]
        confidence = conf.item()
        
        # 2. Severity Estimation (Weighted Average)
        class_weights = torch.arange(1, NUM_CLASSES + 1, device=DEVICE, dtype=torch.float)
        severity_score = torch.sum(probs * class_weights)

    return {
        "class": pred_class,
        "confidence": round(confidence, 4),
        "severity_score": round(severity_score.item(), 4),
        "status": "success"
    }
