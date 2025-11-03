import torch
from torchvision import transforms
from PIL import Image
from model_loader import load_model # Assume model_loader.py is in the same directory
import torch.nn.functional as F

# 1. CONFIGURATION (Adjust based on your final model config)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NOTE: The model path is now relative to the API's file system
MODEL_PATH = "model_trained/hybrid_densenet_inception_resse_final.pth" 

# IMPORTANT: You must provide the actual list of class names here, 
# as reading from a training path (TRAIN_PATH) is bad practice for a live API.
# Replace the list below with your actual class names (e.g., ['healthy', 'disease_a', 'disease_b', ...])
CLASS_NAMES = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 
               'Class_6', 'Class_7', 'Class_8', 'Class_9', 'Class_10', 'Class_11', 'Class_12', 'Class_13']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 299

# 2. MODEL AND TRANSFORM LOADING
# The model file MUST be in a folder named 'model_trained' in your backend repo.
try:
    model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Handle case where model file is not present yet

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_crop_and_severity(image_bytes):
    """Takes image bytes, returns prediction and severity."""
    if model is None:
        return {"error": "Model not loaded. Check deployment files."}, 500
        
    img = Image.open(image_bytes).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        
        # 1. Prediction (Class & Confidence)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        pred_class = CLASS_NAMES[pred_idx.item()]
        confidence = conf.item()
        
        # 2. Severity Estimation
        # This severity calculation assumes your classes are ordered from least to most severe (1 to N)
        severity_score = torch.sum(probs * torch.arange(1, NUM_CLASSES + 1, device=DEVICE, dtype=torch.float))

    return {
        "class": pred_class,
        "confidence": round(confidence, 4),
        "severity_score": round(severity_score.item(), 4),
        "status": "success"
    }
