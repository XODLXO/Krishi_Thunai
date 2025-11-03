from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predictor import predict_crop_and_severity
from io import BytesIO

app = FastAPI(title="Krishi Thunai ML API")

# Setup CORS to allow your GitHub Pages frontend to access the API
# IMPORTANT: Replace "*" with your actual GitHub Pages URL later for security 
# (e.g., "https://yourusername.github.io/Krishi-Thunai-Frontend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ML API is running", "model_loaded": True if predict_crop_and_severity(None).get("status") != "error" else False}

@app.post("/predict")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """API endpoint to receive an image and return prediction and severity."""
    try:
        # Read the image content from the upload
        image_bytes = await file.read()
        image_stream = BytesIO(image_bytes)
        
        # Call the prediction function
        result = predict_crop_and_severity(image_stream)
        
        if result.get("status") == "error":
             raise HTTPException(status_code=500, detail=result.get("error"))

        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
