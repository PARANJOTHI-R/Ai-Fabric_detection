"""
FastAPI Test Client runner verifying /api/health, /api/predict, and /api/camera-frame.
"""

import os
import sys
import cv2
import numpy as np
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.main import app, startup_event

# Run startup event to load models
startup_event()
client = TestClient(app)

print("=" * 65)
print("TESTING FASTAPI ENDPOINTS WITH TESTCLIENT")
print("=" * 65)

# 1. Health check
res = client.get("/api/health")
print(f"GET /api/health: status={res.status_code}, data={res.json()}")
assert res.status_code == 200
assert res.json()["status"] == "ok"

# 2. Predict on clean image (defect free)
clean_path = os.path.join(REPO_ROOT, "Fabric Defects Dataset", "defect free")
clean_files = os.listdir(clean_path)
clean_img_path = os.path.join(clean_path, clean_files[0])
with open(clean_img_path, "rb") as f:
    res = client.post("/api/predict", files={"file": ("clean.jpg", f, "image/jpeg")})
data = res.json()
print(f"\nPOST /api/predict (Clean Fabric):")
print(f"  Status={res.status_code}, QualityStatus={data['quality_status']}, PrimaryDefect={data['primary_defect']}, Conf={data['confidence']}, Latency={data['processing_time_ms']}ms")
assert res.status_code == 200
assert data["quality_status"] == "PASS"

# 3. Predict on defective image (hole)
hole_path = os.path.join(REPO_ROOT, "Fabric Defects Dataset", "hole")
hole_files = os.listdir(hole_path)
hole_img_path = os.path.join(hole_path, hole_files[0])
with open(hole_img_path, "rb") as f:
    res = client.post("/api/predict", files={"file": ("hole.jpg", f, "image/jpeg")})
data = res.json()
print(f"\nPOST /api/predict (Defective Fabric):")
print(f"  Status={res.status_code}, QualityStatus={data['quality_status']}, PrimaryDefect={data['primary_defect']}, Conf={data['confidence']}, Source={data['detection_source']}, Latency={data['processing_time_ms']}ms")
assert res.status_code == 200
assert data["quality_status"] == "DEFECTIVE"

# 4. Predict on blank / dark image
dark_img = np.zeros((480, 640, 3), dtype=np.uint8)
_, dark_buf = cv2.imencode(".jpg", dark_img)
res = client.post("/api/predict", files={"file": ("dark.jpg", dark_buf.tobytes(), "image/jpeg")})
data = res.json()
print(f"\nPOST /api/predict (All-Black / Dark image):")
print(f"  Status={res.status_code}, QualityStatus={data['quality_status']}, PrimaryDefect={data['primary_defect']}, Warning={data['warning']}")
assert res.status_code == 200
assert data["quality_status"] == "STANDBY"

# 5. Camera frame endpoint (Live camera dark frame)
res = client.post("/api/camera-frame", files={"file": ("frame.jpg", dark_buf.tobytes(), "image/jpeg")})
data = res.json()
print(f"\nPOST /api/camera-frame (Dark frame):")
print(f"  Status={res.status_code}, QualityStatus={data['quality_status']}, Status_code={data['status_code']}, Latency={data['processing_time_ms']}ms")
assert res.status_code == 200
assert data["quality_status"] == "STANDBY"

# 6. Camera frame endpoint with real fabric frame
with open(clean_img_path, "rb") as f:
    res = client.post("/api/camera-frame", files={"file": ("frame.jpg", f, "image/jpeg")})
data = res.json()
print(f"\nPOST /api/camera-frame (Live Fabric frame):")
print(f"  Status={res.status_code}, QualityStatus={data['quality_status']}, Brightness={data['mean_brightness']}, Latency={data['processing_time_ms']}ms")
assert res.status_code == 200
assert data["quality_status"] == "PASS"

print("\n" + "=" * 65)
print("ALL FASTAPI ENDPOINT CHECKS PASSED SUCCESSFULLY!")
print("=" * 65)
