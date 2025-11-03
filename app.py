# app.py (Revised Content)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predictor import predict_crop_and_severity
from io import BytesIO

app = FastAPI(title="Krishi Thunai ML API")

# Setup CORS (remains the same)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    # ... (remains the same) ...
    return {"status": "ML API is running", "model_loaded": True if predict_crop_and_severity(BytesIO(b'')).get("status") != "error" else False}

@app.post("/predict")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """API endpoint to receive an image and return primary prediction and severity."""
    try:
        image_bytes = await file.read()
        image_stream = BytesIO(image_bytes)
        
        result = predict_crop_and_severity(image_stream)
        
        if result.get("status") == "error":
             raise HTTPException(status_code=500, detail=result.get("error"))

        # NOTE: The result object now only contains 'class', 'confidence', and 'severity_score'
        return result 
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
