from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Defect(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBox

class PredictionResponse(BaseModel):
    quality_status: str
    defects: List[Defect]
    defect_count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
