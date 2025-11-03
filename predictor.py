# predictor.py

import torch
from torchvision import transforms
from PIL import Image
from model_loader import load_model 
from io import BytesIO

# 1. CONFIGURATION 
# Use CPU by default for cost-effective deployment unless a GPU instance is purchased
DEVICE = torch.device("cpu") 
MODEL_PATH = "model_trained/hybrid_densenet_inception_resse_final.pth" 

# IMPORTANT: Replace these with your 13 actual class names in the correct order
CLASS_NAMES = [
    'Class_1_Healthy', 'Class_2_Mild', 'Class_3_Moderate', 'Class_4_Severe', 
    'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9', 
    'Class_10', 'Class_11', 'Class_12', 'Class_13_Deadliest'
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 299

# 2. MODEL AND TRANSFORM LOADING
model = None
try:
    # Ensure the 'model_trained' folder and the .pth file exist in the deployed environment
    model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    model.eval()
    MODEL_LOAD_STATUS = "success"
except Exception as e:
    print(f"Error loading model from disk: {e}")
    MODEL_LOAD_STATUS = f"error: Model file not found or load failed - {e}"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # Standard ImageNet normalization values
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_crop_and_severity(image_bytes: BytesIO):
    """Takes image bytes (BytesIO), returns prediction and severity."""
    if MODEL_LOAD_STATUS != "success":
        return {"error": MODEL_LOAD_STATUS, "status": "error"}
        
    try:
        img = Image.open(image_bytes).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {e}", "status": "error"}

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        
        # 1. Prediction (Class & Confidence)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        
        pred_class = CLASS_NAMES[pred_idx.item()]
        confidence = conf.item()
        
        # 2. Severity Estimation (Weighted Average)
        # Severity score is a weighted sum: (Prob_Class_1 * 1) + (Prob_Class_2 * 2) + ...
        # This gives a single float score representing average severity level (1.0 to 13.0)
        class_weights = torch.arange(1, NUM_CLASSES + 1, device=DEVICE, dtype=torch.float)
        severity_score = torch.sum(probs * class_weights)

    return {
        "class": pred_class,
        "confidence": round(confidence, 4),
        "severity_score": round(severity_score.item(), 4),
        "status": "success"
    }
