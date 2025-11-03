# backend/core/predictor.py
import torch
from torchvision import transforms
from PIL import Image
from .model_loader import load_model 

# ========= CONFIG & PATHS =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- CORRECT LOCAL PATH ---
MODEL_PATH = "models/hybrid_densenet_inception_resse_final.pth" 

# --- ACTION REQUIRED: REPLACE with your 13+ actual, ordered class names ---
CLASS_NAMES = [
    "class_1_name", 
    "class_2_name", 
    # ... fill in all your class names here! ...
]
NUM_CLASSES = len(CLASS_NAMES)
# -------------------------------------------------------------------------

# ========= LOAD MODEL (Runs on server startup) =========
try:
    print(f"Loading prediction model from: {MODEL_PATH}...")
    model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    model.eval()
    print("Prediction model loaded successfully.")
except Exception as e:
    print(f"Error loading prediction model: {e}. Check models/ folder.")
    model = None 

# ========= IMAGE TRANSFORM =========
IMG_SIZE = 299
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(user_image):
    if model is None:
        raise Exception("Prediction Model not loaded.")
        
    img = user_image.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        pred_class = CLASS_NAMES[pred_idx.item()]
        confidence = conf.item()

    return {
        "class": pred_class,
        "confidence": round(confidence, 4)
    }