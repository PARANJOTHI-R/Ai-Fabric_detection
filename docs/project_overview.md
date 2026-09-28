# Project Overview: AI-Based Fabric Defect Detection

## Problem Statement
Manual inspection in the textile manufacturing pipeline is notoriously slow, subject to human fatigue, and often inconsistent. Minor defects are frequently missed, leading to compromised final product quality and significant financial losses.

## Existing System
Currently, quality checks are either fully manual or rely on traditional computer vision algorithms (like thresholding or edge detection) which are brittle to lighting changes and fabric texture variations.

## Proposed System
An AI-driven Computer Vision system utilizing Deep Learning (YOLO Object Detection) to identify and localize fabric defects in real-time. It operates robustly regardless of minor texture or lighting changes.

## Objectives
1. Automatically detect defects in fabric images.
2. Provide bounding box localization and confidence scores.
3. Classify defects into specific categories.
4. Display results through an intuitive industrial dashboard.

## Dataset
- **Name**: TILDA-400
- **Size**: 400 images
- **Classes**: Hole, Objects, Oil Spot, Thread Error
- **Format**: YOLO format bounding box annotations

## Dataset Preparation
A programmatic pipeline shuffles and splits the data into 80% Training and 20% Validation sets, organizing them into standard YOLO hierarchy and auto-generating the `data.yaml` configuration file.

## Methodology
1. **Preprocessing**: Images are scaled to 640x640 for the neural network.
2. **Inference**: YOLOv8 extracts spatial features and proposes bounding boxes.
3. **NMS**: Non-Maximum Suppression removes duplicate boxes.
4. **Postprocessing**: Predictions are mapped to class names and served via REST API.

## System Architecture
- **Frontend Layer**: React + Tailwind CSS Dashboard
- **Backend Layer**: FastAPI (Python)
- **Model Layer**: Ultralytics YOLO inference engine

## Technology Stack
- **AI Framework**: PyTorch / Ultralytics
- **Web API**: FastAPI
- **User Interface**: React, Vite, Tailwind CSS

## YOLO Model
YOLOv8 nano (`yolov8n.pt`) is chosen for its balance of high mean Average Precision (mAP) and extreme computational efficiency, allowing it to run smoothly on CPUs or edge devices.

## Training
The model is fine-tuned on the TILDA-400 dataset. FAST MODE allows quick epochs (15-20) for prototyping and validation, optimizing for speed and proof-of-concept demonstrations.

## Evaluation
The model is evaluated on the 20% validation split to generate mAP50, mAP50-95, precision, and recall metrics, avoiding data leakage. 

## Actual Results
*(Will be updated once the model finishes training on the specific workspace dataset)*

## Demo Workflow
1. User uploads a fabric image via the React Dashboard.
2. Frontend sends multipart/form-data to FastAPI `POST /api/predict`.
3. FastAPI decodes the image via OpenCV and runs YOLO inference.
4. FastAPI returns a JSON array of defects.
5. React draws bounding boxes over the image and updates the UI status.

## Limitations
- Model is strictly trained on TILDA-400 textures; highly varied fabrics might require retraining.
- High-resolution microscopic defects may not be detected at 640x640 input resolution.

## Future Scope
- **Real-Time Camera Integration**: Direct video feed inspection on weaving machines.
- **Defect Severity**: Classifying severity based on defect surface area.
- **Automated Alerts**: Email or SMS triggers for critical defects.
- **Edge Deployment**: Running on Jetson Nano or similar hardware for factory floors.
