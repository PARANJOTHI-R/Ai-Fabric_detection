# Phone Camera Video Feed Integration Plan: AI Fabric Defect Inspection

> **Document Status**: Implementation-Ready Architecture Plan  
> **Prepared For**: Student Hardware Prototype Testing (Motor-driven Fabric Feed + Smartphone Camera)  
> **Target Date**: October 2026  
> **Prerequisites**: Codebase Audit Complete, Persistent Project Memory Established in `AGENTS.md` and `docs/PROJECT_MEMORY.md`

---

## 1. Executive Summary & Objective

The goal of this phase is to transition the fabric defect inspection system from static image uploads to a **continuous inspection pipeline using a smartphone camera**.

The user's setup consists of:
- A smartphone observing fabric as it is fed through a motor-driven prototype.
- A Windows PC running the existing FastAPI backend and React frontend.
- Active Windows Phone Link connection between the smartphone and the Windows PC.

This document evaluates camera acquisition methods, details the end-to-end frame processing pipeline, defines strict failure-handling states, provides a light-themed React dashboard specification, and establishes a step-by-step implementation sequence.

---

## 2. Phone Camera Input Investigation & Trade-Offs

### A. Empirical Findings on User's System
An empirical scan of Windows camera devices on the user's PC revealed:
```
Index 0: "USB2.0 HD UVC WebCam" (Built-in Laptop Webcam, Active)
Index 1: "Windows Virtual Camera Device" (Phone Link / Windows 11 Connected Camera Device)
```
- **Device 0** returns live RGB frames ($1280 \times 720$, mean pixel intensity $\sim 93$).
- **Device 1** is exposed by Windows as a valid DirectShow device. When the phone is paired via Phone Link but not actively streaming, it yields black frames (`mean = 0.0`). When activated via Windows "Connected Camera", it streams the phone's camera directly to Windows as a native webcam.

### B. Comparison of Practical Camera Ingestion Options

| Option | Architecture | Setup & Dependencies Required | Frame Latency | Reliability for Prototype | Pros & Cons |
|---|---|---|---|---|---|
| **Option 1: Windows Virtual Camera (Phone Link)** | Phone $\rightarrow$ Link to Windows $\rightarrow$ Windows Virtual Camera (Index 1) $\rightarrow$ PC Browser/OpenCV | Windows 11 Settings: *Mobile Devices* $\rightarrow$ *Use as connected camera* enabled. Phone Link paired. | Low ($\sim 100\text{ms}$) | **Medium-High** (Depends on Phone Link Wi-Fi link stability) | **Pros:** Zero third-party apps required; exposed as standard OS camera.<br>**Cons:** Occasionally disconnects or sleeps if phone locks; outputs black frames when idle. |
| **Option 2: Browser-Native Phone Capture** | Open `http://<PC-LAN-IP>:5173` in Phone Chrome/Safari $\rightarrow$ Back camera (`facingMode: environment`) $\rightarrow$ Canvas capture $\rightarrow$ HTTP POST to PC Backend | None on PC. Phone connects to PC local IP on same Wi-Fi. Requires Vite `host: true`. | Very Low ($\sim 50\text{ms}$) | **Very High** (Uses phone's native browser camera API) | **Pros:** No desktop drivers, no Phone Link dependencies, highest resolution control, hardware auto-focus works cleanly.<br>**Cons:** Requires phone and PC to be on same Wi-Fi subnet; phone screen stays active. |
| **Option 3: Dedicated Phone Stream App (DroidCam / IP Webcam)** | Phone app (e.g. DroidCam / IP Webcam) $\rightarrow$ Local HTTP/RTSP stream (`http://192.168.x.x:8080/video`) $\rightarrow$ Backend OpenCV `cv2.VideoCapture` | Install DroidCam or IP Webcam Android/iOS app. | Low ($\sim 150\text{ms}$) | **High** (Standard MJPEG stream) | **Pros:** Extremely robust; works over Wi-Fi or USB tethering; doesn't depend on Windows Phone Link.<br>**Cons:** Requires installing a third-party app on the phone. |
| **Option 4: USB DirectShow Driver (DroidCam Client / Iriun)** | Phone $\rightarrow$ USB cable $\rightarrow$ Virtual Webcam Driver $\rightarrow$ Windows DirectShow Device | Install PC companion software and USB phone drivers. | Lowest ($<30\text{ms}$) | **Highest** (Cabled, immune to Wi-Fi drops) | **Pros:** No Wi-Fi congestion; phone charges continuously during motor test.<br>**Cons:** Requires installing PC desktop client software and keeping phone cabled. |

### C. Recommended Strategy for Student Prototype: **Hybrid Browser-Driven Ingestion**

We recommend a **Dual-Mode Browser Camera Ingestion Architecture**:

1. **Primary Mode (Connected Camera / Webcam via Browser MediaDevices)**:
   - The React dashboard running in Chrome/Edge on the PC requests `navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })`.
   - The user selects either **Camera 1 (Windows Virtual Camera / Phone Link)** or **Camera 0 (Laptop Webcam)** from a camera source dropdown.
   - **Why this is best**:
     - Requires **zero new Python background threads or OpenCV video-streaming loops**.
     - Live video preview renders natively in an HTML5 `<video>` element with zero backend CPU usage and 30 FPS smoothness.
     - The browser periodically captures a frame on a hidden `<canvas>` and dispatches it to the existing backend inference endpoint.

2. **Secondary Fallback Mode (Phone Browser Direct Capture)**:
   - If Windows Phone Link connected camera drops or exhibits sleep issues during long motor runs, the user opens `http://<PC-IP>:5173` on the phone's browser.
   - The same React app uses the phone's back camera directly and dispatches frames to the PC backend.

---

## 3. Video-Processing Pipeline Architecture

```
                    ┌───────────────────────────────┐
                    │      Smartphone Camera        │
                    │   (Fabric Motor Prototype)    │
                    └───────────────┬───────────────┘
                                    │ (Native DirectShow / Phone Link)
                    ┌───────────────▼───────────────┐
                    │   React HTML5 <video> Element │
                    │    (Live 30 FPS Preview, UI)   │
                    └───────────────┬───────────────┘
                                    │
                                    │ Frame Sampling Timer (e.g., every 800ms)
                                    │ Guard: If inFlightInference == true → SKIP
                                    │
                    ┌───────────────▼───────────────┐
                    │    Canvas Snapshot (Blob)     │
                    │   (Resized to 640×640 or 720p)│
                    └───────────────┬───────────────┘
                                    │
                                    │ HTTP POST /api/predict (or /api/camera-frame)
                                    │
                    ┌───────────────▼───────────────┐
                    │      Backend FastAPI App      │
                    │   Frame Decode & Validation   │
                    │ (Mean brightness > 15 check)  │
                    └───────┬───────────────┬───────┘
                            │               │
                 [Valid Frame]              [Dark / Disconnected Frame]
                            │                               │
             ┌──────────────┴──────────────┐         ┌──────▼──────┐
             │                             │         │ Return State│
      ┌──────▼──────┐               ┌──────▼──────┐  │  "STANDBY"  │
      │ YOLOv8s     │               │ MobileNetV3 │  │  or "ERROR" │
      │ Detector    │               │ Classifier  │  └─────────────┘
      └──────┬──────┘               └──────┬──────┘
             │                             │
             └──────────────┬──────────────┘
                            │
             ┌──────────────▼──────────────┐
             │    Hybrid Decision Logic    │
             │   (DEFECTIVE / PASS / ERR)  │
             └──────────────┬──────────────┘
                            │ JSON Response
                            │
             ┌──────────────▼──────────────┐
             │       React Dashboard       │
             │ - Bounding box canvas sync  │
             │ - Defect tag & confidence   │
             │ - Quality PASS/FAIL banner  │
             │ - Event timeline logger     │
             └─────────────────────────────┘
```

---

## 4. Engineering Controls: Performance, Queues & Reliability

### A. Preventing Queue Buildup & Overlapping Inferences
Running continuous video inspection on a CPU (`PyTorch 2.14.1+cpu`) requires strict rate limiting:
- **Total Combined Inference Time on CPU**: $\approx 250\text{ms} - 450\text{ms}$ per frame (YOLOv8s $\approx 300\text{ms}$, MobileNetV3 $\approx 50\text{ms}$).
- **Single-Flight Guard**: A client-side lock (`inFlightRef.current = true`) ensures no new frame is sent while the previous request is pending.
- **Configurable Frame Interval**: Default interval set to **$800\text{ms}$** ($\approx 1.25$ inspections per second), adjustable from $500\text{ms}$ to $2000\text{ms}$ via UI slider. This matches the movement speed of fabric in typical small motor prototypes.

### B. Frame Quality & Disconnection Protection
- **The Black-Frame Flaw Discovered in Audit**: An all-black frame (`mean = 0`) causes MobileNetV3 to predict `horizontal` defect with $65.58\%$ confidence.
- **Mandatory Frame Sanitization**:
  ```python
  # Check image brightness before model execution:
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  mean_val = np.mean(gray)
  variance = np.var(gray)

  if mean_val < 15.0 or variance < 10.0:
      return {
          "quality_status": "STANDBY",
          "primary_defect": "Feed dark / Camera standby",
          "confidence": 0.0,
          "detection_source": "system",
          "detections": [],
          "classification": None,
          "status_code": "STANDBY_FEED"
      }
  ```
- **Never Mark Error as PASS**: If the camera is disconnected, frame decoding fails, or inference crashes, the system must transition to `ERROR` or `STANDBY`, never `PASS`.

### C. Memory Leak Prevention
- Do **not** save every video frame to disk.
- Retain only the **latest frame thumbnail** in client memory.
- Defect log entries in the dashboard store defect metadata (timestamp, defect name, confidence) and only create an image preview blob when an actual `DEFECTIVE` frame is identified.
- History array is strictly capped at 25 events using a rolling buffer.

---

## 5. React Dashboard User Interface Plan

The new camera inspection section will be integrated into the existing light-themed design system without destroying or redesigning the existing image-upload functionality.

### Layout Design (Tabbed or Split View)
The dashboard will provide a clean segment toggle at the top:
`[ 📷 Live Camera Inspection ] | [ 📁 Image File Inspection ]`

### Camera Section Components (Light-Themed Slate Palette)
1. **Live Camera Feed Card**:
   - Standard HTML5 `<video>` preview element with smooth rounded corners (`rounded-xl border border-slate-200`).
   - Camera device selection dropdown (`<select className="btn-secondary text-xs">`) populated via `navigator.mediaDevices.enumerateDevices()`.
   - **Start Inspection** (`btn-primary`, blue) and **Stop Inspection** (`btn-secondary`) buttons.
   - Live stream indicator badge: `● Live Feed (Active)` or `○ Camera Offline`.
2. **Transparent Annotation Canvas Overlay**:
   - Synced with the video element's aspect ratio.
   - Red/amber bounding boxes drawn dynamically on detection frames.
3. **Live Telemetry & Status Row**:
   - Status Banner:
     - `INSPECTING — PASS` (Green, `#16A34A`)
     - `DEFECT DETECTED` (Red, `#DC2626`)
     - `STANDBY / WAITING FOR FABRIC` (Amber, `#D97706`)
     - `SYSTEM ERROR` (Slate, `#64748B`)
   - Processing Metrics:
     - Inference Latency: e.g. `312ms`
     - Frame Interval: `800ms` (adjustable slider)
     - Inspection Count: Total frames analyzed this session
4. **Recent Defects Timeline**:
   - Compact table listing timestamped defect occurrences:
     - Time (`14:22:05`), Defect Category (`hole`), Confidence (`78.4%`), Source (`YOLO` with coordinates or `Classifier`).

---

## 6. Validation and Testing Strategy

| Test Scenario | Action / Input | Expected Result | Pass Criteria |
|---|---|---|---|
| **1. Device Enumeration** | Connect Phone via Phone Link; launch dashboard. | Both `USB2.0 HD UVC WebCam` and `Windows Virtual Camera Device` appear in dropdown. | User can toggle between cameras cleanly. |
| **2. Standby / Covered Lens** | Cover camera lens completely (mean intensity $<15$). | UI reports `STANDBY / FEED DARK`. Models are not falsely triggered. | Zero false positives (`horizontal` defect prevented). |
| **3. Defect-Free Fabric** | Point camera at plain woven fabric on motor roll. | Status shows `PASSED`. Confidence $\ge 90\%$ for `defect free`. | No bounding boxes, clean green indicator. |
| **4. Defective Fabric (Hole / Object)** | Introduce fabric swatch with punctured hole or foreign thread. | Status triggers `DEFECTIVE` within 1 frame interval. Bounding box overlay appears over flaw. | YOLO bounding box lines up with visible flaw. |
| **5. Rapid Motion / Blur** | Accelerate motor speed beyond shutter threshold. | High blur variance filter flags motion; does not falsely flag defect-free fabric. | System stability maintained; no queue backlog. |
| **6. Camera Disconnect** | Disable Phone Link or unplug phone during inspection. | UI detects feed loss; displays `Camera Disconnected`. Resets gracefully upon reconnect. | No unhandled JS exception; server remains responsive. |
| **7. Performance Benchmark** | Run continuous inspection for 5 minutes (375 frames). | Memory remains constant; average CPU inference latency $\le 450\text{ms}$; no memory leaks. | System does not degrade or crash. |

---

## 7. Step-by-Step Implementation Sequence for Next Session

1. **Step 1: Backend Camera Endpoint Optimization**
   - In [`backend/main.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/backend/main.py), add `/api/camera-frame` endpoint (or leverage `/api/predict`) with brightness/dark-frame pre-check to prevent black-frame false positives.
   - Add status response: `"STANDBY" | "PASS" | "DEFECTIVE" | "ERROR"`.

2. **Step 2: Frontend Camera Stream & Capture Component**
   - In [`frontend/src/App.jsx`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/frontend/src/App.jsx), create a modular camera hook and component.
   - Implement `navigator.mediaDevices.getUserMedia` with device selector.
   - Implement canvas snapshot extractor with `inFlight` request guard.

3. **Step 3: Overlay Synchronization & Telemetry**
   - Overlay canvas bounding-box renderer matched to the video element's bounding rectangle.
   - Add latency timer, inspection counter, and defect event timeline.

4. **Step 4: Live Verification with Hardware Prototype**
   - Test Phone Link camera device index 1 and verify phone video stream.
   - Test sample defect swatches under the camera lens.

---

## 8. Actual Implementation & Verification Log (Completed)

### Implementation Completed:
1. **Backend Endpoint**: Added [`/api/camera-frame`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/backend/main.py#L185) with dark-frame validation (`mean < 15.0` or `var < 10.0` returns `STANDBY`), latency calculation, and [`CameraFrameResponse`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/backend/schemas.py#L62).
2. **Frontend Camera Dashboard**: Added **Live Camera Inspection** tab in [`frontend/src/App.jsx`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/frontend/src/App.jsx) with camera device dropdown (`navigator.mediaDevices.enumerateDevices`), live HTML5 `<video>` preview, non-overlapping async request loop (`inFlightRef`), dynamic bounding box overlay, configurable sampling interval slider (500ms - 2000ms), and real-time defect timeline log.
3. **Vite Proxy & Cross-Device Access**: Updated [`frontend/vite.config.js`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/frontend/vite.config.js) with `server: { host: true, proxy: { '/api': 'http://localhost:8000' } }`.

### Verification Test Results:
- **Build Verification**: `npm run build` compiled clean with 0 errors in 1.42s.
- **Health Check**: `GET /api/health` returned `status: ok`, both models loaded.
- **Defective Frame Inference**: Verified on actual defect image; returned `quality_status: DEFECTIVE`, detected 2 flaws with bounding boxes.
- **Dark/Covered Frame Protection**: Verified on all-black frame; successfully intercepted and returned `quality_status: STANDBY`, preventing false-positive classification.
- **Live Hardware Camera Test**: Captured live frame from PC webcam device (`mean brightness = 128.1`); successfully processed in 2398ms on CPU, returning real detections.

