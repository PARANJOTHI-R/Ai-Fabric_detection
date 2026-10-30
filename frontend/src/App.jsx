import React, { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import {
  UploadCloud, CheckCircle, AlertTriangle, Activity,
  Image as ImageIcon, RefreshCw, Scan, Grid3X3,
  ChevronRight, Clock, TrendingUp, Package, Layers,
  XCircle, Search, Loader2, Cpu, Tag, Camera,
  Video, VideoOff, Play, Square, Sliders, Gauge, Sun
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

// ─── Fabric inspection SVG logo ───────────────────────────────────────────────
const FabricLogo = () => (
  <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="32" height="32" rx="8" fill="#2563EB" fillOpacity="0.1"/>
    <rect x="5" y="8" width="22" height="16" rx="2" stroke="#2563EB" strokeWidth="1.5" fill="none"/>
    <line x1="9" y1="8" x2="9" y2="24" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="13" y1="8" x2="13" y2="24" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="17" y1="8" x2="17" y2="24" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="21" y1="8" x2="21" y2="24" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="5" y1="12" x2="27" y2="12" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="5" y1="16" x2="27" y2="16" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="5" y1="20" x2="27" y2="20" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <circle cx="22" cy="22" r="6" fill="#2563EB"/>
    <path d="M19.5 22L21.5 24L24.5 20.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// ─── Class colour map (YOLO TILDA-400 + Classifier classes) ──────────────────
const CLASS_COLORS = {
  // YOLO / TILDA-400
  'hole':         { border: '#DC2626', bg: 'rgba(220,38,38,0.15)',  text: '#DC2626' },
  'objects':      { border: '#D97706', bg: 'rgba(217,119,6,0.15)',  text: '#D97706' },
  'oil spot':     { border: '#7C3AED', bg: 'rgba(124,58,237,0.15)', text: '#7C3AED' },
  'thread error': { border: '#DB2777', bg: 'rgba(219,39,119,0.15)', text: '#DB2777' },
  // Classifier (new dataset)
  'stain':        { border: '#9333EA', bg: 'rgba(147,51,234,0.15)', text: '#9333EA' },
  'lines':        { border: '#0891B2', bg: 'rgba(8,145,178,0.15)',  text: '#0891B2' },
  'horizontal':   { border: '#EA580C', bg: 'rgba(234,88,12,0.15)',  text: '#EA580C' },
  'vertical':     { border: '#65A30D', bg: 'rgba(101,163,13,0.15)', text: '#65A30D' },
  'Vertical':     { border: '#65A30D', bg: 'rgba(101,163,13,0.15)', text: '#65A30D' },
  'defect free':  { border: '#16A34A', bg: 'rgba(22,163,74,0.15)',  text: '#16A34A' },
};

const getClassColor = (name) =>
  CLASS_COLORS[name?.toLowerCase()] ||
  CLASS_COLORS[name] ||
  { border: '#DC2626', bg: 'rgba(220,38,38,0.15)', text: '#DC2626' };

// ─── Source badge ─────────────────────────────────────────────────────────────
const SourceBadge = ({ source }) => {
  if (source === 'yolo' || source === 'both')
    return (
      <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-600 border border-blue-200">
        <Scan className="w-3 h-3" /> YOLO Detection
      </span>
    );
  if (source === 'classifier')
    return (
      <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-purple-50 text-purple-600 border border-purple-200">
        <Cpu className="w-3 h-3" /> AI Classifier
      </span>
    );
  if (source === 'system')
    return (
      <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-50 text-amber-600 border border-amber-200">
        <Activity className="w-3 h-3" /> System Check
      </span>
    );
  return null;
};

// ─── Confidence bar ───────────────────────────────────────────────────────────
const ConfBar = ({ value }) => {
  if (value == null) return null;
  const pct = (value * 100).toFixed(1);
  const color = value >= 0.6 ? '#DC2626' : value >= 0.35 ? '#D97706' : '#64748B';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-border overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-semibold tabular-nums w-12 text-right" style={{ color }}>{pct}%</span>
    </div>
  );
};

// ─── Stat card ────────────────────────────────────────────────────────────────
const StatCard = ({ icon: Icon, label, value, sub, iconColor }) => (
  <div className="card p-4 flex items-start gap-3">
    <div className="p-2 rounded-lg bg-surface-2 shrink-0">
      <Icon className="w-4 h-4" style={{ color: iconColor || '#2563EB' }} />
    </div>
    <div className="min-w-0">
      <p className="text-xs text-secondary font-medium mb-0.5">{label}</p>
      <p className="text-lg font-bold text-dark truncate">{value}</p>
      {sub && <p className="text-xs text-muted mt-0.5">{sub}</p>}
    </div>
  </div>
);

// ─── History thumbnail ────────────────────────────────────────────────────────
const HistoryItem = ({ item, onClick, isActive }) => {
  const isPass = item.result.quality_status === 'PASS';
  const isStandby = item.result.quality_status === 'STANDBY';
  return (
    <button
      onClick={() => onClick(item)}
      className={`history-card w-full text-left ${isActive ? 'border-primary bg-primary-light/30' : ''}`}
    >
      <div className="w-10 h-10 rounded-lg overflow-hidden bg-surface-2 border border-border shrink-0">
        <img src={item.previewUrl} alt="" className="w-full h-full object-cover" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-dark truncate">{item.filename}</p>
        <p className="text-xs text-muted mt-0.5">{item.timestamp}</p>
      </div>
      <span className={`badge shrink-0 ${isPass ? 'badge-success' : isStandby ? 'badge-warning bg-amber-50 text-amber-700 border-amber-200' : 'badge-danger'}`}>
        {isPass ? 'PASS' : isStandby ? 'STANDBY' : 'FAIL'}
      </span>
    </button>
  );
};

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState('camera'); // 'camera' | 'upload'

  // ── Upload Mode States ──
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [dragging, setDragging] = useState(false);
  const [history, setHistory] = useState([]);
  const [activeHistoryId, setActiveHistoryId] = useState(null);

  const imageRef = useRef(null);
  const fileInputRef = useRef(null);

  // ── Camera Mode States ──
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const [isInspecting, setIsInspecting] = useState(false);
  const [sampleInterval, setSampleInterval] = useState(800); // ms
  const [cameraResult, setCameraResult] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [cameraDefectsHistory, setCameraDefectsHistory] = useState([]);
  const [cameraFramesAnalyzed, setCameraFramesAnalyzed] = useState(0);
  const [lastProcessingMs, setLastProcessingMs] = useState(0);
  const [lastBrightness, setLastBrightness] = useState(null);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const inFlightRef = useRef(false);
  const isInspectingRef = useRef(false);
  const timerIdRef = useRef(null);
  const frameSeqRef = useRef(0);
  const latestProcessedSeqRef = useRef(0);
  const rollingWindowRef = useRef([]);
  const lastDefectLogRef = useRef({ defect: null, time: 0 });

  // ── Camera Device Enumeration ───────────────────────────────────────────────
  const enumerateCameras = useCallback(async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        setCameraError('MediaDevices API is not supported in this browser.');
        return;
      }
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices.filter(d => d.kind === 'videoinput');
      setDevices(videoInputs);

      if (videoInputs.length > 0) {
        setSelectedDeviceId(prev => {
          if (prev && videoInputs.some(d => d.deviceId === prev)) return prev;
          // Prefer Phone Link / Virtual Camera device if available
          const preferred = videoInputs.find(d =>
            d.label.toLowerCase().includes('virtual') ||
            d.label.toLowerCase().includes('phone') ||
            d.label.toLowerCase().includes('link')
          );
          return preferred ? preferred.deviceId : videoInputs[0].deviceId;
        });
      }
    } catch (err) {
      console.error('Error enumerating devices:', err);
      setCameraError('Failed to enumerate cameras: ' + err.message);
    }
  }, []);

  useEffect(() => {
    enumerateCameras();
    return () => {
      stopInspection();
      stopCamera();
    };
  }, [enumerateCameras]);

  // ── Camera Control Functions ───────────────────────────────────────────────
  const startCamera = async (overrideDeviceId) => {
    setCameraError(null);
    const targetId = overrideDeviceId !== undefined ? overrideDeviceId : selectedDeviceId;

    // Stop existing stream if active
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    try {
      const constraints = {
        video: targetId
          ? { deviceId: { exact: targetId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(e => console.warn('Video play warning:', e));
      }

      setCameraActive(true);
      // Re-enumerate to get device labels now that permissions are granted
      enumerateCameras();

      // Handle stream termination / unplugging
      stream.getVideoTracks().forEach(track => {
        track.onended = () => {
          stopCamera();
          setCameraError('Camera stream disconnected or ended by system.');
        };
      });

    } catch (err) {
      console.error('Failed to open camera:', err);
      setCameraActive(false);
      setCameraError(
        `Unable to access camera (${err.name}: ${err.message}). ` +
        `Ensure camera permissions are allowed and no other app is locking the feed.`
      );
    }
  };

  const stopCamera = () => {
    stopInspection();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setCameraResult(null);
    rollingWindowRef.current = [];
  };

  // ── Frame Sampling & Non-Overlapping Inference ──────────────────────────────
  const captureAndInspectFrame = async () => {
    if (!videoRef.current || !streamRef.current || inFlightRef.current || !isInspectingRef.current) return;
    const video = videoRef.current;
    if (video.readyState < 2 || video.videoWidth === 0) {
      if (isInspectingRef.current) {
        timerIdRef.current = setTimeout(captureAndInspectFrame, 200);
      }
      return;
    }

    inFlightRef.current = true;
    const frameSeq = ++frameSeqRef.current;
    const t0 = performance.now();

    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
      if (!blob || !isInspectingRef.current) {
        inFlightRef.current = false;
        return;
      }

      const formData = new FormData();
      formData.append('file', blob, 'camera_frame.jpg');

      const response = await axios.post(`${API_BASE}/api/camera-frame`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 6000
      });

      // Discard stale response if a newer frame already completed or inspection stopped
      if (frameSeq < latestProcessedSeqRef.current || !isInspectingRef.current) {
        return;
      }
      latestProcessedSeqRef.current = frameSeq;

      const res = response.data;
      res._frameWidth = canvas.width;
      res._frameHeight = canvas.height;
      res._seqId = frameSeq;

      // Temporal Rolling Window (last 3 valid frames) for Result Stability
      if (res.quality_status === 'STANDBY' || res.quality_status === 'ERROR') {
        rollingWindowRef.current = [];
        setCameraResult(res);
      } else {
        // Strong defect: YOLO confidence >= 0.25 or Classifier >= 0.70
        const isStrongDefect = res.quality_status === 'DEFECTIVE' && (
          (res.detections && res.detections.some(d => d.confidence >= 0.25)) ||
          (res.confidence >= 0.70)
        );

        rollingWindowRef.current.push({
          status: res.quality_status,
          defect: res.primary_defect,
          conf: res.confidence,
          isStrong: isStrongDefect
        });
        if (rollingWindowRef.current.length > 3) {
          rollingWindowRef.current.shift();
        }

        const recent = rollingWindowRef.current;
        let stabilizedStatus = res.quality_status;
        let stabilizedDefect = res.primary_defect;

        if (isStrongDefect) {
          // Strong defect: alert immediately without delay
          stabilizedStatus = 'DEFECTIVE';
        } else if (res.quality_status === 'DEFECTIVE') {
          // Weak defect: require corroboration in at least 2 of last 3 frames
          const defectCountInWindow = recent.filter(r => r.status === 'DEFECTIVE').length;
          if (defectCountInWindow < 2 && recent.length >= 2) {
            stabilizedStatus = 'PASS';
            stabilizedDefect = null;
          }
        } else {
          // Frame is PASS: if at least 2 of last 3 frames are PASS, confirm PASS
          const passCount = recent.filter(r => r.status === 'PASS').length;
          if (passCount >= 2 || recent.length === 1) {
            stabilizedStatus = 'PASS';
            stabilizedDefect = null;
          }
        }

        const finalResult = {
          ...res,
          quality_status: stabilizedStatus,
          primary_defect: stabilizedDefect,
          detections: stabilizedStatus === 'PASS' ? [] : res.detections
        };

        setCameraResult(finalResult);

        // History Logging: Log defect once every 3s if persistent, or immediately if new
        if (stabilizedStatus === 'DEFECTIVE') {
          const nowMs = Date.now();
          const lastLog = lastDefectLogRef.current;
          const defectName = stabilizedDefect || 'Defect';
          const isNewDefect = lastLog.defect !== defectName;
          const isDebounced = (nowMs - lastLog.time) >= 3000;

          if (isNewDefect || isDebounced) {
            lastDefectLogRef.current = { defect: defectName, time: nowMs };
            const now = new Date();
            const event = {
              id: nowMs,
              timestamp: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              defect: defectName,
              confidence: res.confidence,
              source: res.detection_source,
              detectionsCount: finalResult.detections?.length || 0,
            };
            setCameraDefectsHistory(prev => [event, ...prev.slice(0, 24)]);
          }
        } else if (stabilizedStatus === 'PASS') {
          lastDefectLogRef.current = { defect: null, time: 0 };
        }
      }

      const totalElapsedMs = Math.round(performance.now() - t0);
      setCameraFramesAnalyzed(prev => prev + 1);
      setLastProcessingMs(res.processing_time_ms || totalElapsedMs);
      setLastBrightness(res.mean_brightness ?? null);

    } catch (err) {
      console.error('Frame inspection error:', err);
    } finally {
      inFlightRef.current = false;
      // Schedule next frame with controlled sampleInterval strictly after completion
      if (isInspectingRef.current && cameraActive) {
        timerIdRef.current = setTimeout(captureAndInspectFrame, sampleInterval);
      }
    }
  };

  const startInspection = () => {
    if (!cameraActive) {
      setCameraError('Please start the camera before initiating inspection.');
      return;
    }
    isInspectingRef.current = true;
    setIsInspecting(true);
    setCameraError(null);
    captureAndInspectFrame();
  };

  const stopInspection = () => {
    isInspectingRef.current = false;
    setIsInspecting(false);
    inFlightRef.current = false;
    if (timerIdRef.current) {
      clearTimeout(timerIdRef.current);
      timerIdRef.current = null;
    }
    setCameraResult(null); // Clear obsolete bounding boxes immediately
    rollingWindowRef.current = [];
  };

  // Sync inspection loop lifecycle with cameraActive state
  useEffect(() => {
    if (!cameraActive && isInspectingRef.current) {
      stopInspection();
    }
    return () => {
      if (timerIdRef.current) {
        clearTimeout(timerIdRef.current);
        timerIdRef.current = null;
      }
    };
  }, [cameraActive]);

  // ── Upload Mode File handling ──────────────────────────────────────────────
  const loadFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please select a valid image file (JPG, PNG, JPEG).');
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setActiveHistoryId(null);
  }, []);

  const handleFileSelect = (e) => loadFile(e.target.files[0]);
  const handleDragOver  = (e) => { e.preventDefault(); setDragging(true); };
  const handleDragLeave = ()  => setDragging(false);
  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    loadFile(e.dataTransfer.files[0]);
  };

  const resetImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setActiveHistoryId(null);
  };

  const inspectFabric = async () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE}/api/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const res = response.data;
      setResult(res);

      const now = new Date();
      const entry = {
        id: Date.now(),
        filename: selectedFile.name,
        previewUrl,
        result: res,
        timestamp: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setHistory(prev => [entry, ...prev.slice(0, 19)]);
      setActiveHistoryId(entry.id);
    } catch (err) {
      console.error(err);
      setError(
        err.response?.data?.detail ||
        'Inspection failed. Please ensure the backend server is running on port 8000.'
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const loadHistoryItem = (item) => {
    setPreviewUrl(item.previewUrl);
    setResult(item.result);
    setSelectedFile(null);
    setError(null);
    setActiveHistoryId(item.id);
    setImageSize({ width: 0, height: 0 });
  };

  // ── Bounding Box Overlays ───────────────────────────────────────────────────
  const renderUploadBoundingBoxes = () => {
    if (!result || !result.detections || result.detections.length === 0) return null;
    if (!imageSize.width || !imageRef.current) return null;

    const { clientWidth: dw, clientHeight: dh } = imageRef.current;
    if (dw === 0 || dh === 0) return null;
    const scaleX = dw / imageSize.width;
    const scaleY = dh / imageSize.height;

    return result.detections.map((det, i) => {
      const { x1, y1, x2, y2 } = det.bbox;
      const color = getClassColor(det.class_name);
      // Strictly clamp coordinates to rendered image boundaries
      const clampedX1 = Math.max(0, Math.min(dw, x1 * scaleX));
      const clampedY1 = Math.max(0, Math.min(dh, y1 * scaleY));
      const clampedX2 = Math.max(0, Math.min(dw, x2 * scaleX));
      const clampedY2 = Math.max(0, Math.min(dh, y2 * scaleY));
      const boxW = Math.max(0, clampedX2 - clampedX1);
      const boxH = Math.max(0, clampedY2 - clampedY1);

      return (
        <div
          key={i}
          className="absolute z-10 pointer-events-none"
          style={{
            left: `${clampedX1}px`,
            top:  `${clampedY1}px`,
            width:  `${boxW}px`,
            height: `${boxH}px`,
            border: `2px solid ${color.border}`,
            backgroundColor: color.bg,
          }}
        >
          <div
            className="absolute -top-6 left-0 text-white text-xs font-bold px-1.5 py-0.5 rounded whitespace-nowrap shadow-sm"
            style={{ backgroundColor: color.border }}
          >
            {det.class_name.toUpperCase()} {(det.confidence * 100).toFixed(0)}%
          </div>
        </div>
      );
    });
  };

  const renderCameraBoundingBoxes = () => {
    if (!cameraResult || !cameraResult.detections || cameraResult.detections.length === 0) return null;
    if (!videoRef.current) return null;

    const { clientWidth: dw, clientHeight: dh } = videoRef.current;
    if (dw === 0 || dh === 0) return null;

    const srcW = cameraResult._frameWidth || videoRef.current.videoWidth || videoSize.width || 640;
    const srcH = cameraResult._frameHeight || videoRef.current.videoHeight || videoSize.height || 480;

    // Calculate exact rendered video rectangle inside <video> element accounting for CSS object-contain
    const scale = Math.min(dw / srcW, dh / srcH);
    const renderedW = srcW * scale;
    const renderedH = srcH * scale;
    const offsetX = (dw - renderedW) / 2;
    const offsetY = (dh - renderedH) / 2;

    return cameraResult.detections.map((det, i) => {
      const { x1, y1, x2, y2 } = det.bbox;
      const color = getClassColor(det.class_name);
      // Strictly clamp coordinates to rendered video content boundaries
      const clampedX1 = offsetX + Math.max(0, Math.min(srcW, x1)) * scale;
      const clampedY1 = offsetY + Math.max(0, Math.min(srcH, y1)) * scale;
      const clampedX2 = offsetX + Math.max(0, Math.min(srcW, x2)) * scale;
      const clampedY2 = offsetY + Math.max(0, Math.min(srcH, y2)) * scale;
      const boxW = Math.max(0, clampedX2 - clampedX1);
      const boxH = Math.max(0, clampedY2 - clampedY1);

      return (
        <div
          key={i}
          className="absolute z-10 pointer-events-none transition-all duration-150"
          style={{
            left: `${clampedX1}px`,
            top:  `${clampedY1}px`,
            width:  `${boxW}px`,
            height: `${boxH}px`,
            border: `2.5px solid ${color.border}`,
            backgroundColor: color.bg,
          }}
        >
          <div
            className="absolute -top-6 left-0 text-white text-xs font-bold px-1.5 py-0.5 rounded whitespace-nowrap shadow-sm"
            style={{ backgroundColor: color.border }}
          >
            {det.class_name.toUpperCase()} {(det.confidence * 100).toFixed(0)}%
          </div>
        </div>
      );
    });
  };

  // ── Statistics Calculations ────────────────────────────────────────────────
  const totalScanned = history.length;
  const totalDefective = history.filter(h => h.result.quality_status === 'DEFECTIVE').length;
  const passRate = totalScanned > 0
    ? ((1 - totalDefective / totalScanned) * 100).toFixed(0)
    : '--';

  const isUploadPass = result?.quality_status === 'PASS';
  const isUploadStandby = result?.quality_status === 'STANDBY';
  const isUploadError = result?.quality_status === 'ERROR';
  const isUploadDefective = result?.quality_status === 'DEFECTIVE';
  const hasUploadYoloDetections = result?.detections?.length > 0;
  const hasUploadClassifierOnly = result?.detection_source === 'classifier' && !hasUploadYoloDetections;

  const isCameraPass = cameraResult?.quality_status === 'PASS';
  const isCameraStandby = cameraResult?.quality_status === 'STANDBY';
  const isCameraError = cameraResult?.quality_status === 'ERROR';
  const isCameraDefective = cameraResult?.quality_status === 'DEFECTIVE';
  const hasCameraYoloDetections = cameraResult?.detections?.length > 0;
  const hasCameraClassifierOnly = cameraResult?.detection_source === 'classifier' && !hasCameraYoloDetections;

  return (
    <div className="min-h-screen bg-background flex flex-col">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="bg-surface border-b border-border sticky top-0 z-50 shadow-card">
        <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FabricLogo />
            <div>
              <h1 className="font-bold text-base text-dark leading-none">Fabric Defect Inspection</h1>
              <p className="text-xs text-secondary font-medium mt-0.5">AI Quality Inspection · Hybrid Pipeline</p>
            </div>
          </div>

          {/* Mode Switcher Tabs */}
          <div className="flex items-center gap-1.5 p-1 bg-surface-2 rounded-xl border border-border">
            <button
              onClick={() => setActiveTab('camera')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'camera'
                  ? 'bg-primary text-white shadow-sm'
                  : 'text-secondary hover:text-dark hover:bg-surface'
              }`}
            >
              <Camera className="w-3.5 h-3.5" />
              Live Camera
              {isInspecting && (
                <span className="w-2 h-2 rounded-full bg-danger animate-pulse" />
              )}
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'upload'
                  ? 'bg-primary text-white shadow-sm'
                  : 'text-secondary hover:text-dark hover:bg-surface'
              }`}
            >
              <UploadCloud className="w-3.5 h-3.5" />
              Image Upload
            </button>
          </div>
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────────────────── */}
      <main className="flex-grow max-w-screen-xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">

        {/* ── Stats Row ──────────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          {activeTab === 'camera' ? (
            <>
              <StatCard
                icon={Scan}
                label="Frames Inspected"
                value={cameraFramesAnalyzed}
                sub={isInspecting ? `Sampling every ${sampleInterval}ms` : 'Inspection paused'}
                iconColor="#2563EB"
              />
              <StatCard
                icon={AlertTriangle}
                label="Live Defects Caught"
                value={cameraDefectsHistory.length}
                sub="Logged this session"
                iconColor="#DC2626"
              />
              <StatCard
                icon={Gauge}
                label="Inference Latency"
                value={lastProcessingMs > 0 ? `${lastProcessingMs} ms` : '--'}
                sub="CPU execution time"
                iconColor="#D97706"
              />
              <StatCard
                icon={Activity}
                label="Stream State"
                value={isInspecting ? 'ACTIVE' : cameraActive ? 'STANDBY' : 'OFFLINE'}
                sub={cameraActive ? 'Camera connected' : 'No feed'}
                iconColor={isInspecting ? '#16A34A' : '#64748B'}
              />
            </>
          ) : (
            <>
              <StatCard icon={Scan}          label="Inspected"  value={totalScanned}             sub="This session"      iconColor="#2563EB" />
              <StatCard icon={CheckCircle}   label="Passed"     value={totalScanned - totalDefective} sub="No defects found" iconColor="#16A34A" />
              <StatCard icon={AlertTriangle} label="Defective"  value={totalDefective}           sub="Defects detected"  iconColor="#DC2626" />
              <StatCard icon={TrendingUp}    label="Pass Rate"  value={passRate === '--' ? '--' : `${passRate}%`} sub="Current session" iconColor="#D97706" />
            </>
          )}
        </div>

        {/* ══════════════════════════════════════════════════════════════════════
            TAB 1: LIVE CAMERA INSPECTION (PHONE CAMERA / WEBCAM)
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'camera' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">

            {/* ── Left Sidebar: Controls & Devices ────────────────────────── */}
            <aside className="lg:col-span-3 flex flex-col gap-3">
              <div className="card p-4">
                <h2 className="text-sm font-semibold text-dark flex items-center gap-2 mb-3">
                  <Video className="w-4 h-4 text-primary" />
                  Camera Feed Setup
                </h2>

                {/* Device Selector */}
                <div className="space-y-3">
                  <div>
                    <label className="text-xs font-semibold text-secondary uppercase tracking-wider block mb-1">
                      Camera Source
                    </label>
                    <select
                      value={selectedDeviceId}
                      onChange={(e) => {
                        const id = e.target.value;
                        setSelectedDeviceId(id);
                        if (cameraActive) startCamera(id);
                      }}
                      className="w-full text-xs bg-surface-2 border border-border rounded-lg px-2.5 py-2 font-medium text-dark focus:outline-none focus:border-primary truncate"
                    >
                      {devices.length === 0 ? (
                        <option value="">Default System Camera</option>
                      ) : (
                        devices.map((d, idx) => {
                          const isPhoneLink = d.label.toLowerCase().includes('virtual') ||
                                              d.label.toLowerCase().includes('phone') ||
                                              d.label.toLowerCase().includes('link');
                          return (
                            <option key={d.deviceId || idx} value={d.deviceId}>
                              {d.label ? (isPhoneLink ? `📱 ${d.label}` : `📷 ${d.label}`) : `Camera ${idx + 1}`}
                            </option>
                          );
                        })
                      )}
                    </select>
                    <p className="text-[11px] text-muted mt-1 leading-snug">
                      Select <strong>Windows Virtual Camera</strong> for Phone Link, or laptop webcam.
                    </p>
                  </div>

                  {/* Sampling Interval Slider */}
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-xs font-semibold text-secondary uppercase tracking-wider">
                        Sampling Rate
                      </label>
                      <span className="text-xs font-bold text-primary tabular-nums">{sampleInterval} ms</span>
                    </div>
                    <input
                      type="range"
                      min="500"
                      max="2000"
                      step="100"
                      value={sampleInterval}
                      onChange={(e) => setSampleInterval(Number(e.target.value))}
                      className="w-full accent-primary h-1.5 bg-border rounded-lg cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-muted mt-0.5">
                      <span>Fast (500ms)</span>
                      <span>Balanced (800ms)</span>
                      <span>Slow (2s)</span>
                    </div>
                  </div>

                  {/* Camera Action Buttons */}
                  <div className="pt-2 flex flex-col gap-2">
                    {!cameraActive ? (
                      <button
                        onClick={() => startCamera()}
                        className="btn-primary w-full justify-center text-xs py-2"
                      >
                        <Video className="w-4 h-4" />
                        Start Camera Stream
                      </button>
                    ) : (
                      <div className="grid grid-cols-2 gap-2">
                        {!isInspecting ? (
                          <button
                            onClick={startInspection}
                            className="btn-primary justify-center text-xs py-2 bg-success hover:bg-green-700"
                          >
                            <Play className="w-3.5 h-3.5" />
                            Start Inspection
                          </button>
                        ) : (
                          <button
                            onClick={stopInspection}
                            className="btn-primary justify-center text-xs py-2 bg-danger hover:bg-red-700"
                          >
                            <Square className="w-3.5 h-3.5" />
                            Stop Inspection
                          </button>
                        )}
                        <button
                          onClick={stopCamera}
                          className="btn-secondary justify-center text-xs py-2"
                        >
                          <VideoOff className="w-3.5 h-3.5 text-danger" />
                          Stop Feed
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Recent Defects Caught on Feed */}
              <div className="card p-4 flex-1">
                <h3 className="text-xs font-semibold text-secondary uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Clock className="w-3.5 h-3.5" />
                  Live Flaw Events ({cameraDefectsHistory.length})
                </h3>
                {cameraDefectsHistory.length === 0 ? (
                  <div className="py-6 text-center text-muted">
                    <CheckCircle className="w-6 h-6 text-success mx-auto mb-1.5 opacity-60" />
                    <p className="text-xs">No defects captured yet.</p>
                    <p className="text-[11px] text-muted">Defects appear here as fabric moves.</p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-[300px] overflow-y-auto custom-scrollbar pr-1">
                    {cameraDefectsHistory.map(item => (
                      <div key={item.id} className="p-2.5 rounded-lg border border-border bg-surface-2 text-xs">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-bold capitalize text-danger flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-danger animate-pulse" />
                            {item.defect}
                          </span>
                          <span className="text-[11px] text-muted tabular-nums">{item.timestamp}</span>
                        </div>
                        <div className="flex justify-between items-center text-secondary">
                          <span>Confidence: {(item.confidence * 100).toFixed(0)}%</span>
                          <SourceBadge source={item.source} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </aside>

            {/* ── Center: Live Camera Feed with Overlay ────────────────────── */}
            <div className="lg:col-span-5 flex flex-col gap-3">
              <div className="card p-4 flex flex-col min-h-[480px]">
                <div className="flex justify-between items-center mb-3">
                  <h2 className="text-sm font-semibold text-dark flex items-center gap-2">
                    <Camera className="w-4 h-4 text-primary" />
                    Fabric Feed Preview
                  </h2>
                  <div className="flex items-center gap-2">
                    {cameraActive && (
                      <span className="inline-flex items-center gap-1.5 text-xs text-success px-2.5 py-1 bg-success-light rounded-full border border-success/20 font-medium">
                        <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
                        {isInspecting ? 'Inspecting Live' : 'Streaming'}
                      </span>
                    )}
                    {videoSize.width > 0 && (
                      <span className="text-xs text-muted tabular-nums">
                        {videoSize.width}×{videoSize.height}
                      </span>
                    )}
                  </div>
                </div>

                {/* Error Banner */}
                {cameraError && (
                  <div className="bg-danger-light border border-danger/20 rounded-lg p-3 flex items-start gap-2 mb-3 animate-fade-in">
                    <AlertTriangle className="w-4 h-4 text-danger shrink-0 mt-0.5" />
                    <p className="text-xs text-danger leading-relaxed">{cameraError}</p>
                  </div>
                )}

                {/* Camera Container */}
                <div className="flex-grow bg-slate-950 rounded-xl border border-border overflow-hidden flex items-center justify-center min-h-[380px] relative">
                  {!cameraActive ? (
                    <div className="text-center p-6 text-slate-400 flex flex-col items-center gap-3">
                      <div className="p-4 bg-slate-900 rounded-full border border-slate-800">
                        <Camera className="w-10 h-10 text-slate-500" />
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-slate-200">Camera Feed Inactive</p>
                        <p className="text-xs text-slate-400 mt-1 max-w-xs">
                          Click <strong>Start Camera Stream</strong> to activate your phone camera or webcam.
                        </p>
                      </div>
                      <button
                        onClick={() => startCamera()}
                        className="btn-primary text-xs mt-1"
                      >
                        <Play className="w-3.5 h-3.5" />
                        Activate Camera
                      </button>
                    </div>
                  ) : (
                    <div className="w-full h-full flex items-center justify-center relative p-2">
                      <div className="relative inline-block leading-none">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          muted
                          className="max-h-[420px] max-w-full object-contain select-none block rounded-lg"
                          onLoadedMetadata={(e) => {
                            setVideoSize({
                              width: e.target.videoWidth,
                              height: e.target.videoHeight
                            });
                          }}
                        />
                        {/* Bounding box dynamic overlays anchored directly to video frame */}
                        {renderCameraBoundingBoxes()}
                      </div>

                      {/* Standby / Dark Frame Notice */}
                      {isCameraStandby && (
                        <div className="absolute bottom-3 left-3 right-3 bg-amber-950/80 backdrop-blur border border-amber-500/40 text-amber-200 p-2.5 rounded-lg flex items-center gap-2 text-xs">
                          <Sun className="w-4 h-4 text-amber-400 shrink-0" />
                          <span>Standby / Feed dark. Point phone camera directly at illuminated fabric.</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Telemetry pill under video */}
                {cameraActive && (
                  <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-muted px-2 py-1.5 bg-surface-2 rounded-lg border border-border">
                    <span className="truncate">
                      Brightness: <strong className="text-dark">{lastBrightness != null ? `${lastBrightness} avg` : '--'}</strong>
                    </span>
                    <span className="truncate">
                      Latency: <strong className="text-dark">{lastProcessingMs > 0 ? `${lastProcessingMs} ms` : '--'}</strong>
                    </span>
                    <span className="truncate">
                      Sample: <strong className="text-dark">{sampleInterval} ms</strong>
                    </span>
                    <span className="truncate">
                      Effective: <strong className="text-primary font-semibold">{lastProcessingMs > 0 ? `${(1000 / (lastProcessingMs + sampleInterval)).toFixed(1)} FPS` : '--'}</strong>
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* ── Right: Live Frame Inspection Results ──────────────────────── */}
            <div className="lg:col-span-4 flex flex-col gap-3">
              <div className="card p-4 flex-1">
                <h2 className="text-sm font-semibold text-dark flex items-center gap-2 mb-4">
                  <Activity className="w-4 h-4 text-primary" />
                  Live Quality Status
                </h2>

                {!cameraResult ? (
                  <div className="py-12 text-center text-muted flex flex-col items-center gap-3">
                    <div className="p-4 bg-surface-2 rounded-xl">
                      <Scan className="w-10 h-10 text-muted opacity-50" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-secondary">Awaiting Live Frames</p>
                      <p className="text-xs text-muted mt-1 leading-relaxed">
                        Start the camera and click <strong>Start Inspection</strong><br />
                        to inspect fabric passing under the lens.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4 animate-slide-up">
                    {/* Status Banner */}
                    <div className={`rounded-xl p-4 flex items-center gap-3 ${
                      isCameraPass
                        ? 'status-pass'
                        : isCameraDefective
                        ? 'status-defective'
                        : isCameraStandby
                        ? 'bg-amber-50 border border-amber-200 text-amber-700'
                        : 'bg-slate-100 border border-slate-200 text-slate-700'
                    }`}>
                      <div>
                        {isCameraPass && <CheckCircle className="w-9 h-9 text-success" />}
                        {isCameraDefective && <AlertTriangle className="w-9 h-9 text-danger" />}
                        {isCameraStandby && <Sun className="w-9 h-9 text-amber-500" />}
                        {isCameraError && <XCircle className="w-9 h-9 text-slate-600" />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold uppercase tracking-wider opacity-70 mb-0.5">
                          Fabric Condition
                        </p>
                        <p className="text-2xl font-black tracking-tight">
                          {cameraResult.quality_status}
                        </p>
                      </div>
                      {cameraResult.detection_source && (
                        <SourceBadge source={cameraResult.detection_source} />
                      )}
                    </div>

                    {/* Image Quality Warning Advisory */}
                    {cameraResult.warning && (
                      <div className="bg-amber-50 border border-amber-200 text-amber-800 rounded-lg p-3 flex items-start gap-2 text-xs">
                        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                        <span>{cameraResult.warning}</span>
                      </div>
                    )}

                    {/* PASS state description */}
                    {isCameraPass && (
                      <div className="bg-surface-2 rounded-xl p-4 text-center border border-border">
                        <CheckCircle className="w-7 h-7 text-success mx-auto mb-1.5" />
                        <p className="font-semibold text-dark text-sm">Fabric Defect-Free</p>
                        <p className="text-xs text-secondary mt-0.5">Weave continuity meets quality standards.</p>
                        {cameraResult.classification && (
                          <div className="mt-2.5 pt-2.5 border-t border-border text-xs text-muted">
                            Classifier: <span className="font-semibold text-dark capitalize">{cameraResult.classification.class_name}</span>
                            {' '}({(cameraResult.classification.confidence * 100).toFixed(1)}%)
                          </div>
                        )}
                      </div>
                    )}

                    {/* DEFECTIVE state details */}
                    {isCameraDefective && (
                      <>
                        <div
                          className="rounded-lg p-3 border"
                          style={{
                            borderColor: `${getClassColor(cameraResult.primary_defect).border}40`,
                            backgroundColor: getClassColor(cameraResult.primary_defect).bg
                          }}
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <Tag className="w-3.5 h-3.5" style={{ color: getClassColor(cameraResult.primary_defect).text }} />
                            <span className="text-xs font-semibold uppercase tracking-wider text-secondary">Detected Flaw</span>
                          </div>
                          <p className="text-xl font-black capitalize" style={{ color: getClassColor(cameraResult.primary_defect).text }}>
                            {cameraResult.primary_defect}
                          </p>
                          <ConfBar value={cameraResult.confidence} />
                        </div>

                        {/* YOLO Detections with coordinates */}
                        {hasCameraYoloDetections && (
                          <div>
                            <div className="flex items-center justify-between mb-2">
                              <p className="text-xs font-semibold text-secondary uppercase tracking-wider flex items-center gap-1.5">
                                <Scan className="w-3.5 h-3.5" /> YOLO Bounding Boxes
                              </p>
                              <span className="badge badge-danger">{cameraResult.detections.length} flaw(s)</span>
                            </div>
                            <div className="space-y-1.5 max-h-[220px] overflow-y-auto custom-scrollbar">
                              {cameraResult.detections.map((det, i) => {
                                const c = getClassColor(det.class_name);
                                return (
                                  <div key={i} className="rounded-lg p-2.5 border text-xs" style={{ borderColor: `${c.border}30`, backgroundColor: c.bg }}>
                                    <div className="flex justify-between items-center font-bold capitalize text-dark mb-1">
                                      <span>{det.class_name}</span>
                                      <span style={{ color: c.border }}>{(det.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="text-[11px] text-muted grid grid-cols-2">
                                      <span>X: {det.bbox.x1} → {det.bbox.x2}</span>
                                      <span>Y: {det.bbox.y1} → {det.bbox.y2}</span>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Classifier-Only note */}
                        {hasCameraClassifierOnly && (
                          <div className="p-3 bg-surface-2 rounded-lg border border-border text-xs text-muted">
                            <p className="font-medium text-dark capitalize mb-1">
                              Flaw: {cameraResult.classification?.class_name} ({(cameraResult.confidence * 100).toFixed(1)}%)
                            </p>
                            ℹ️ Texture classifier defect — whole image flagged (no bounding box available).
                          </div>
                        )}
                      </>
                    )}

                    {/* STANDBY state message */}
                    {isCameraStandby && (
                      <div className="bg-amber-50 rounded-xl p-4 text-center border border-amber-200">
                        <Sun className="w-7 h-7 text-amber-500 mx-auto mb-1.5" />
                        <p className="font-semibold text-dark text-sm">Waiting for Fabric</p>
                        <p className="text-xs text-secondary mt-1">
                          The camera feed is currently dark or obstructed. Place illuminated fabric beneath the lens.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════════
            TAB 2: IMAGE FILE UPLOAD INSPECTION (PRESERVED ORIGINAL FUNCTIONALITY)
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'upload' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">

            {/* ── Left: History + Model Info ──────────────────────────────── */}
            <aside className="lg:col-span-3 flex flex-col gap-3">
              <div className="card p-4 flex-shrink-0">
                <h2 className="text-sm font-semibold text-dark flex items-center gap-2 mb-3">
                  <Clock className="w-4 h-4 text-secondary" />
                  Recent Inspections
                  {history.length > 0 && (
                    <span className="ml-auto badge badge-primary">{history.length}</span>
                  )}
                </h2>

                {history.length === 0 ? (
                  <div className="py-8 text-center">
                    <Grid3X3 className="w-8 h-8 text-muted mx-auto mb-2 opacity-50" />
                    <p className="text-xs text-muted">No inspections yet.</p>
                    <p className="text-xs text-muted">Upload an image to start.</p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-[420px] overflow-y-auto custom-scrollbar pr-1">
                    {history.map(item => (
                      <HistoryItem
                        key={item.id}
                        item={item}
                        onClick={loadHistoryItem}
                        isActive={item.id === activeHistoryId}
                      />
                    ))}
                  </div>
                )}
              </div>

              {/* Model Info Card */}
              <div className="card p-4">
                <h3 className="text-xs font-semibold text-secondary uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Layers className="w-3.5 h-3.5" />
                  Model Info
                </h3>
                <div className="space-y-2 text-xs">
                  {[
                    { label: 'Detection',    value: 'YOLOv8s (50 epochs)' },
                    { label: 'Classifier',   value: 'MobileNetV3-Small' },
                    { label: 'YOLO Classes', value: '4 defect types' },
                    { label: 'CLF Classes',  value: '6 fabric classes' },
                    { label: 'Threshold',    value: 'YOLO 0.06–0.15 | CLF 0.45 / 0.50 cumulative' },
                  ].map(({ label, value }) => (
                    <div key={label} className="flex justify-between items-start gap-2">
                      <span className="text-secondary flex-shrink-0">{label}</span>
                      <span className="font-medium text-dark text-right">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </aside>

            {/* ── Center: Image Panel ─────────────────────────────────────── */}
            <div className="lg:col-span-5 flex flex-col gap-3">
              <div className="card p-4 flex flex-col min-h-[480px]">
                <div className="flex justify-between items-center mb-3">
                  <h2 className="text-sm font-semibold text-dark flex items-center gap-2">
                    <ImageIcon className="w-4 h-4 text-secondary" />
                    Fabric Image
                  </h2>
                  {previewUrl && (
                    <button
                      onClick={resetImage}
                      className="text-xs text-secondary hover:text-danger flex items-center gap-1 transition-colors"
                    >
                      <XCircle className="w-3.5 h-3.5" />
                      Clear
                    </button>
                  )}
                </div>

                {!previewUrl ? (
                  <div
                    className={`upload-zone flex-grow p-8 min-h-[360px] ${dragging ? 'dragging' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <div className={`p-4 rounded-full mb-4 transition-transform duration-200 ${dragging ? 'scale-110 bg-primary-light' : 'bg-surface-2'}`}>
                      <UploadCloud className={`w-8 h-8 ${dragging ? 'text-primary' : 'text-muted'}`} />
                    </div>
                    <p className="font-semibold text-dark mb-1">
                      {dragging ? 'Drop to upload' : 'Drop fabric image here'}
                    </p>
                    <p className="text-sm text-secondary mb-4">or click to browse files</p>
                    <span className="btn-secondary text-xs px-3 py-1.5">Browse Files</span>
                    <p className="text-xs text-muted mt-3">Supports JPG, PNG, JPEG</p>
                    <input
                      ref={fileInputRef}
                      id="file-upload"
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handleFileSelect}
                    />
                  </div>
                ) : (
                  <div className="flex-grow flex flex-col gap-3">
                    <div className="flex-grow bg-surface-2 rounded-xl border border-border overflow-hidden flex items-center justify-center min-h-[320px] relative p-2">
                      <div className="relative inline-block">
                        <img
                          ref={imageRef}
                          src={previewUrl}
                          alt="Fabric Preview"
                          className="max-h-[380px] max-w-full object-contain rounded-lg select-none"
                          onLoad={(e) => setImageSize({ width: e.target.naturalWidth, height: e.target.naturalHeight })}
                          draggable={false}
                        />
                        {renderUploadBoundingBoxes()}
                      </div>
                    </div>

                    {selectedFile && (
                      <p className="text-xs text-secondary truncate px-1">
                        <span className="text-muted">File:</span> {selectedFile.name}
                      </p>
                    )}
                  </div>
                )}

                {previewUrl && selectedFile && (
                  <div className="mt-3 flex justify-end">
                    <button
                      onClick={inspectFabric}
                      disabled={isProcessing}
                      className="btn-primary"
                      id="inspect-btn"
                    >
                      {isProcessing ? (
                        <>
                          <Loader2 className="w-4 h-4 spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Search className="w-4 h-4" />
                          Inspect Fabric
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* ── Right: Results Panel ─────────────────────────────────────── */}
            <div className="lg:col-span-4 flex flex-col gap-3">
              <div className="card p-4 flex-1">
                <h2 className="text-sm font-semibold text-dark flex items-center gap-2 mb-4">
                  <Activity className="w-4 h-4 text-secondary" />
                  Inspection Results
                </h2>

                {error && (
                  <div className="bg-danger-light border border-danger/20 rounded-lg p-3 flex items-start gap-2 mb-4 animate-fade-in">
                    <AlertTriangle className="w-4 h-4 text-danger shrink-0 mt-0.5" />
                    <p className="text-xs text-danger leading-relaxed">{error}</p>
                  </div>
                )}

                {!result && !error && !isProcessing && (
                  <div className="py-12 text-center text-muted flex flex-col items-center gap-3">
                    <div className="p-4 bg-surface-2 rounded-xl">
                      <Package className="w-10 h-10 text-muted opacity-50" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-secondary">No Inspection Yet</p>
                      <p className="text-xs text-muted mt-1">Upload a fabric image and click<br />Inspect Fabric to see results.</p>
                    </div>
                  </div>
                )}

                {isProcessing && (
                  <div className="space-y-3 animate-pulse">
                    <div className="h-16 bg-surface-2 rounded-lg" />
                    <div className="h-24 bg-surface-2 rounded-lg" />
                    <div className="h-12 bg-surface-2 rounded-lg" />
                  </div>
                )}

                {result && !isProcessing && (
                  <div className="space-y-4 animate-slide-up">
                    {/* Status Banner */}
                    <div className={`rounded-xl p-4 flex items-center gap-3 ${
                      isUploadPass
                        ? 'status-pass'
                        : isUploadDefective
                        ? 'status-defective'
                        : isUploadStandby
                        ? 'bg-amber-50 border border-amber-200 text-amber-700'
                        : 'bg-slate-100 border border-slate-200 text-slate-700'
                    }`}>
                      <div>
                        {isUploadPass && <CheckCircle className="w-9 h-9 text-success" />}
                        {isUploadDefective && <AlertTriangle className="w-9 h-9 text-danger" />}
                        {isUploadStandby && <Sun className="w-9 h-9 text-amber-500" />}
                        {isUploadError && <XCircle className="w-9 h-9 text-slate-600" />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold uppercase tracking-wider opacity-70 mb-0.5">
                          Overall Quality
                        </p>
                        <p className="text-2xl font-black tracking-tight">
                          {result.quality_status}
                        </p>
                      </div>
                      {result.detection_source && result.detection_source !== 'none' && (
                        <SourceBadge source={result.detection_source} />
                      )}
                    </div>

                    {/* Image Quality Warning Advisory */}
                    {result.warning && (
                      <div className="bg-amber-50 border border-amber-200 text-amber-800 rounded-lg p-3 flex items-start gap-2 text-xs">
                        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                        <span>{result.warning}</span>
                      </div>
                    )}

                    {/* Telemetry row */}
                    {result.processing_time_ms > 0 && (
                      <div className="flex items-center justify-between text-xs text-muted px-1">
                        <span>Latency: {result.processing_time_ms} ms</span>
                        {result.total_defect_probability != null && (
                          <span>P(defect): {(result.total_defect_probability * 100).toFixed(1)}%</span>
                        )}
                      </div>
                    )}

                    {/* STANDBY state */}
                    {isUploadStandby && (
                      <div className="bg-amber-50/60 rounded-xl p-4 text-center border border-amber-200">
                        <Sun className="w-7 h-7 text-amber-500 mx-auto mb-1.5" />
                        <p className="font-semibold text-amber-900 text-sm">Standby / Blank Image</p>
                        <p className="text-xs text-amber-700 mt-0.5">Image is dark or uniform. Point camera or upload a clear, illuminated fabric image.</p>
                      </div>
                    )}

                    {/* PASS state description */}
                    {isUploadPass && (
                      <div className="bg-surface-2 rounded-xl p-5 text-center border border-border">
                        <CheckCircle className="w-8 h-8 text-success mx-auto mb-2" />
                        <p className="font-semibold text-dark text-sm">No Defect Detected</p>
                        <p className="text-xs text-secondary mt-1">The fabric meets quality standards.</p>
                        {result.classification && (
                          <div className="mt-3 pt-3 border-t border-border">
                            <p className="text-xs text-muted">
                              Classifier: <span className="font-semibold text-dark capitalize">{result.classification.class_name}</span>
                              {' '}({(result.classification.confidence * 100).toFixed(1)}%)
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {isUploadDefective && (
                      <>
                        {result.primary_defect && (
                          <div
                            className="rounded-lg p-3 border"
                            style={{
                              borderColor: `${getClassColor(result.primary_defect).border}40`,
                              backgroundColor: getClassColor(result.primary_defect).bg
                            }}
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <Tag className="w-3.5 h-3.5" style={{ color: getClassColor(result.primary_defect).text }} />
                              <span className="text-xs font-semibold uppercase tracking-wider text-secondary">Primary Defect</span>
                            </div>
                            <p className="text-xl font-black capitalize" style={{ color: getClassColor(result.primary_defect).text }}>
                              {result.primary_defect}
                            </p>
                            {result.confidence != null && <ConfBar value={result.confidence} />}
                          </div>
                        )}

                        {hasUploadYoloDetections && (
                          <div>
                            <div className="flex items-center justify-between mb-2 px-0.5">
                              <p className="text-xs font-semibold text-secondary uppercase tracking-wider flex items-center gap-1.5">
                                <Scan className="w-3.5 h-3.5" /> YOLO Detections
                              </p>
                              <span className="badge badge-danger">{result.detections.length} found</span>
                            </div>
                            <div className="space-y-2 max-h-[280px] overflow-y-auto custom-scrollbar">
                              {result.detections.map((det, i) => {
                                const color = getClassColor(det.class_name);
                                return (
                                  <div
                                    key={i}
                                    className="rounded-lg p-3 border transition-all"
                                    style={{ borderColor: `${color.border}30`, backgroundColor: color.bg }}
                                  >
                                    <div className="flex items-start justify-between mb-2">
                                      <div className="flex items-center gap-2">
                                        <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color.border }} />
                                        <span className="font-semibold text-sm capitalize text-dark">{det.class_name}</span>
                                      </div>
                                      <ChevronRight className="w-3.5 h-3.5 text-muted" />
                                    </div>
                                    <ConfBar value={det.confidence} />
                                    <div className="grid grid-cols-2 gap-x-3 mt-2 text-xs text-secondary">
                                      <span>x: {det.bbox.x1} → {det.bbox.x2}</span>
                                      <span>y: {det.bbox.y1} → {det.bbox.y2}</span>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {hasUploadClassifierOnly && result.classification && (
                          <div>
                            <div className="flex items-center justify-between mb-2 px-0.5">
                              <p className="text-xs font-semibold text-secondary uppercase tracking-wider flex items-center gap-1.5">
                                <Cpu className="w-3.5 h-3.5" /> Classifier Result
                              </p>
                            </div>
                            {(() => {
                              const color = getClassColor(result.classification.class_name);
                              return (
                                <div
                                  className="rounded-lg p-3 border"
                                  style={{ borderColor: `${color.border}30`, backgroundColor: color.bg }}
                                >
                                  <div className="flex items-center gap-2 mb-2">
                                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color.border }} />
                                    <span className="font-semibold text-sm capitalize text-dark">
                                      {result.classification.class_name}
                                    </span>
                                  </div>
                                  <ConfBar value={result.classification.confidence} />
                                  <p className="text-xs text-muted mt-2">
                                    ℹ️ Classification result — no bounding box available
                                  </p>
                                </div>
                              );
                            })()}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-border bg-surface py-4">
        <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-muted">
          <p>Fabric Defect Inspection &mdash; AI-Powered Quality Control</p>
          <p>&bull; Live Phone Camera &bull; YOLOv8s + MobileNetV3 Hybrid Pipeline &bull; FastAPI + React 19</p>
        </div>
      </footer>
    </div>
  );
}
