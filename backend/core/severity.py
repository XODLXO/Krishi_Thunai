# backend/core/severity.py
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Imports config and path from predictor.py
from .model_loader import load_model
from .predictor import NUM_CLASSES, MODEL_PATH, DEVICE 

# ========= LOAD MODEL =========
try:
    print(f"Loading severity model from: {MODEL_PATH}...")
    model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    model.eval()
    print("Severity model loaded successfully.")
except Exception as e:
    print(f"Error loading severity model: {e}. Check models/ folder.")
    model = None

# ========= IMAGE TRANSFORM =========
IMG_SIZE = 299
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def estimate_severity(user_image):
    if model is None:
        raise Exception("Severity Model not loaded.")
        
    img = user_image.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        
        # Severity score calculation
        severity_score = torch.sum(probs * torch.arange(1, NUM_CLASSES+1, device=DEVICE, dtype=torch.float))

    return {
        "severity_score": round(severity_score.item(), 4)
    }