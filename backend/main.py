from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .model_service import ModelService
from .preprocessing import preprocess_image
from .schemas import PredictionResponse, HealthResponse
import uvicorn

app = FastAPI(title="AI Fabric Inspection API")

# Setup CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = None

@app.on_event("startup")
def startup_event():
    global model_service
    model_service = ModelService()

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_service.model is not None if model_service else False
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    if not model_service or not model_service.model:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")
    
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
            
        result = model_service.predict(image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
