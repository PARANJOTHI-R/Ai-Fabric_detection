# AI-Based Fabric Defect Detection and Quality Inspection

## Problem Statement
Manual fabric inspection in the textile industry is time-consuming, labor-intensive, subjective, and prone to human error. It is challenging to perform consistently at a large scale, resulting in increased costs and potential quality control escapes.

## Objective
The objective is to build an AI-based computer vision system that automatically analyzes fabric images and detects fabric defects with high accuracy and speed.

## Dataset
- **Name**: TILDA-400
- **Task**: Object Detection
- **Format**: YOLO format annotations
- **Structure**:
  ```
  dataset/
  ├── images/ (train/val)
  ├── labels/ (train/val)
  └── data.yaml
  ```

## Defect Classes
The system detects the following 4 classes:
1. Hole (0)
2. Objects (1)
3. Oil Spot (2)
4. Thread Error (3)

## System Architecture
The project follows a modular client-server architecture:
1. **Frontend**: React.js + Tailwind CSS providing a professional UI.
2. **Backend**: Python FastAPI serving the trained AI model.
3. **AI Engine**: Ultralytics YOLO object detection.

## Technology Stack
- **Deep Learning**: PyTorch, Ultralytics YOLOv8
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend**: React, TailwindCSS, Vite
- **Data Processing**: OpenCV, NumPy, Pandas

## YOLO Model
We use a lightweight YOLO model (`yolov8n`) to ensure fast inference speeds and low compute requirements, making it ideal for edge deployments or local execution without heavy GPUs.

## Dataset Preparation
The custom `prepare_dataset.py` script:
1. Shuffles and splits the TILDA-400 dataset (80% train, 20% val).
2. Converts and moves images and `.txt` labels to the YOLO standard directory structure.
3. Automatically generates the required `data.yaml`.

## Training Process
The `train.py` script leverages Ultralytics APIs to train the YOLO model. It automatically detects GPU availability (CUDA) or falls back to CPU. The model saves weights periodically and tracks performance metrics during training.

## Evaluation and Results
After training, the `evaluate.py` script validates the best weights on the validation dataset. 
*(Once training completes, actual evaluation metrics like mAP50, mAP50-95, precision, and recall will be populated in the `results/evaluation/metrics.json`)*.

## Backend API
The FastAPI backend exposes two main endpoints:
- `GET /api/health`: Checks system status and model readiness.
- `POST /api/predict`: Accepts an uploaded image, runs YOLO inference, calculates defect bounding boxes and confidence scores, and returns structured JSON containing the quality status (PASS/DEFECTIVE).

## Frontend
The React dashboard provides a professional, industrial-grade user interface:
- Drag-and-drop file upload.
- Image preview with drawn YOLO bounding boxes indicating defect locations.
- Real-time prediction display, showing confidence scores and defect details.

## How to Run

### 1. Backend
```bash
pip install -r requirements.txt
python -m backend.main
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

## Example Prediction
**Input**: Fabric image with an oil spot.
**Output**:
```json
{
  "quality_status": "DEFECTIVE",
  "defects": [
    {
      "class_name": "oil spot",
      "confidence": 0.9234,
      "bbox": { "x1": 120, "y1": 80, "x2": 260, "y2": 190 }
    }
  ],
  "defect_count": 1
}
```

## Limitations
- Performance depends strictly on lighting conditions similar to the training dataset.
- Extremely small defects might be missed at lower image resolutions (e.g., <640px).

## Future Enhancements
- Real-time camera inspection.
- Production-line PLC integration.
- Defect severity estimation.
- Higher-resolution edge-device deployment.
