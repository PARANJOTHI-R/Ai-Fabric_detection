import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import {
  UploadCloud, CheckCircle, AlertTriangle, Activity,
  Image as ImageIcon, RefreshCw, Scan, Grid3X3,
  ChevronRight, Clock, TrendingUp, Package, Layers,
  XCircle, Search, Loader2, Cpu, Tag
} from 'lucide-react';

// ─── Fabric inspection SVG logo ───────────────────────────────────────────────
const FabricLogo = () => (
  <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="32" height="32" rx="8" fill="#2563EB" fillOpacity="0.1"/>
    <rect x="5" y="8" width="22" height="16" rx="2" stroke="#2563EB" strokeWidth="1.5" fill="none"/>
    {/* Weave pattern */}
    <line x1="9" y1="8" x2="9" y2="24" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="13" y1="8" x2="13" y2="24" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="17" y1="8" x2="17" y2="24" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="21" y1="8" x2="21" y2="24" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="5" y1="12" x2="27" y2="12" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="5" y1="16" x2="27" y2="16" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="5" y1="20" x2="27" y2="20" stroke="#2563EB" strokeWidth="1" strokeDasharray="2 2"/>
    {/* Scan overlay */}
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
  return null;
};

// ─── Confidence bar ───────────────────────────────────────────────────────────
const ConfBar = ({ value }) => {
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
      <span className={`badge shrink-0 ${isPass ? 'badge-success' : 'badge-danger'}`}>
        {isPass ? 'PASS' : 'FAIL'}
      </span>
    </button>
  );
};

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
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

  // ── File handling ──────────────────────────────────────────────────────────
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

  // ── Inspect ────────────────────────────────────────────────────────────────
  const inspectFabric = async () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/api/predict', formData, {
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

  // ── Load history item ──────────────────────────────────────────────────────
  const loadHistoryItem = (item) => {
    setPreviewUrl(item.previewUrl);
    setResult(item.result);
    setSelectedFile(null);
    setError(null);
    setActiveHistoryId(item.id);
    setImageSize({ width: 0, height: 0 });
  };

  // ── Bounding boxes (YOLO detections ONLY — never classifier results) ───────
  const renderBoundingBoxes = () => {
    if (!result || !result.detections || result.detections.length === 0) return null;
    if (!imageSize.width || !imageRef.current) return null;

    const { clientWidth: dw, clientHeight: dh } = imageRef.current;
    const scaleX = dw / imageSize.width;
    const scaleY = dh / imageSize.height;

    return result.detections.map((det, i) => {
      const { x1, y1, x2, y2 } = det.bbox;
      const color = getClassColor(det.class_name);
      return (
        <div
          key={i}
          className="absolute z-10"
          style={{
            left: `${x1 * scaleX}px`,
            top:  `${y1 * scaleY}px`,
            width:  `${(x2 - x1) * scaleX}px`,
            height: `${(y2 - y1) * scaleY}px`,
            border: `2px solid ${color.border}`,
            backgroundColor: color.bg,
          }}
        >
          <div
            className="absolute -top-6 left-0 text-white text-xs font-bold px-1.5 py-0.5 rounded whitespace-nowrap"
            style={{ backgroundColor: color.border }}
          >
            {det.class_name.toUpperCase()} {(det.confidence * 100).toFixed(0)}%
          </div>
        </div>
      );
    });
  };

  // ── Statistics from history ────────────────────────────────────────────────
  const totalScanned  = history.length;
  const totalDefective = history.filter(h => h.result.quality_status === 'DEFECTIVE').length;
  const passRate = totalScanned > 0
    ? ((1 - totalDefective / totalScanned) * 100).toFixed(0)
    : '--';

  const isPass = result?.quality_status === 'PASS';
  // Does YOLO have real bounding-box detections?
  const hasYoloDetections = result?.detections?.length > 0;
  // Classifier-only result (no bounding boxes to show)
  const hasClassifierOnly =
    result?.detection_source === 'classifier' && !hasYoloDetections;

  return (
    <div className="min-h-screen bg-background flex flex-col">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="bg-surface border-b border-border sticky top-0 z-50 shadow-card">
        <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FabricLogo />
            <div>
              <h1 className="font-bold text-base text-dark leading-none">Fabric Defect Inspection</h1>
              <p className="text-xs text-secondary font-medium mt-0.5">AI-Powered Quality Control · Hybrid Pipeline</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* <span className="hidden sm:flex items-center gap-1.5 text-xs text-secondary px-3 py-1.5 bg-surface-2 rounded-lg border border-border"> */}
              {/* <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse-soft"></span> */}
              {/* System Active */}
            {/* </span> */}
            {/* <span className="text-xs font-medium px-3 py-1.5 bg-primary-light text-primary rounded-lg border border-primary/20"> */}
              {/* Dashboard */}
            {/* </span> */}
          </div>
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────────────────── */}
      <main className="flex-grow max-w-screen-xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">

        {/* ── Stats Row ──────────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          <StatCard icon={Scan}          label="Inspected"  value={totalScanned}             sub="This session"      iconColor="#2563EB" />
          <StatCard icon={CheckCircle}   label="Passed"     value={totalScanned - totalDefective} sub="No defects found" iconColor="#16A34A" />
          <StatCard icon={AlertTriangle} label="Defective"  value={totalDefective}           sub="Defects detected"  iconColor="#DC2626" />
          <StatCard icon={TrendingUp}    label="Pass Rate"  value={passRate === '--' ? '--' : `${passRate}%`} sub="Current session" iconColor="#D97706" />
        </div>

        {/* ── Content Grid ───────────────────────────────────────────────── */}
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

            {/* Model info card */}
            <div className="card p-4">
              <h3 className="text-xs font-semibold text-secondary uppercase tracking-wider mb-3 flex items-center gap-2">
                <Layers className="w-3.5 h-3.5" />
                Model Info
              </h3>
              <div className="space-y-2 text-xs">
                {[
                  { label: 'Detection',    value: 'YOLOv8' },
                  { label: 'Classifier',   value: 'MobileNetV3-Small' },
                  { label: 'YOLO Classes', value: '4 defect types' },
                  { label: 'CLF Classes',  value: '6 fabric classes' },
                  { label: 'Threshold',    value: '0.55 (classifier)' },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between">
                    <span className="text-secondary">{label}</span>
                    <span className="font-medium text-dark">{value}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 pt-3 border-t border-border">
                <div className="text-xs text-secondary mb-2">YOLO Defects</div>
                <div className="flex flex-wrap gap-1">
                  {['hole','objects','oil spot','thread error'].map(name => {
                    const c = getClassColor(name);
                    return (
                      <span key={name} className="text-xs px-2 py-0.5 rounded-full font-medium capitalize"
                        style={{ backgroundColor: c.bg, color: c.text, border: `1px solid ${c.border}30` }}>
                        {name}
                      </span>
                    );
                  })}
                </div>
                <div className="text-xs text-secondary mb-2 mt-2">Classifier Classes</div>
                <div className="flex flex-wrap gap-1">
                  {['stain','lines','horizontal','Vertical','hole','defect free'].map(name => {
                    const c = getClassColor(name);
                    return (
                      <span key={name} className="text-xs px-2 py-0.5 rounded-full font-medium capitalize"
                        style={{ backgroundColor: c.bg, color: c.text, border: `1px solid ${c.border}30` }}>
                        {name}
                      </span>
                    );
                  })}
                </div>
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
                /* Upload zone */
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
                /* Image preview with bounding boxes */
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
                      {/* Only YOLO detections get bounding boxes */}
                      {renderBoundingBoxes()}
                    </div>
                  </div>

                  {selectedFile && (
                    <p className="text-xs text-secondary truncate px-1">
                      <span className="text-muted">File:</span> {selectedFile.name}
                    </p>
                  )}
                </div>
              )}

              {/* Inspect button */}
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

              {/* Error */}
              {error && (
                <div className="bg-danger-light border border-danger/20 rounded-lg p-3 flex items-start gap-2 mb-4 animate-fade-in">
                  <AlertTriangle className="w-4 h-4 text-danger shrink-0 mt-0.5" />
                  <p className="text-xs text-danger leading-relaxed">{error}</p>
                </div>
              )}

              {/* Waiting */}
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

              {/* Loading skeleton */}
              {isProcessing && (
                <div className="space-y-3 animate-pulse">
                  <div className="h-16 bg-surface-2 rounded-lg" />
                  <div className="h-24 bg-surface-2 rounded-lg" />
                  <div className="h-12 bg-surface-2 rounded-lg" />
                  <div className="h-12 bg-surface-2 rounded-lg" />
                </div>
              )}

              {/* Result */}
              {result && !isProcessing && (
                <div className="space-y-4 animate-slide-up">

                  {/* Quality status banner */}
                  <div className={`rounded-xl p-4 flex items-center gap-3 ${isPass ? 'status-pass' : 'status-defective'}`}>
                    <div>
                      {isPass
                        ? <CheckCircle className="w-9 h-9 text-success" />
                        : <AlertTriangle className="w-9 h-9 text-danger" />
                      }
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-semibold uppercase tracking-wider opacity-70 mb-0.5">
                        Overall Quality
                      </p>
                      <p className="text-2xl font-black tracking-tight">
                        {isPass ? 'PASSED' : 'DEFECTIVE'}
                      </p>
                    </div>
                    {result.detection_source && result.detection_source !== 'none' && (
                      <SourceBadge source={result.detection_source} />
                    )}
                  </div>

                  {/* ── PASS: defect free ─────────────────────────────── */}
                  {isPass && (
                    <div className="bg-surface-2 rounded-xl p-5 text-center border border-border">
                      <CheckCircle className="w-8 h-8 text-success mx-auto mb-2" />
                      <p className="font-semibold text-dark text-sm">No Defect Detected</p>
                      <p className="text-xs text-secondary mt-1">The fabric meets quality standards.</p>
                      {/* Show classifier class even on PASS (informational) */}
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

                  {/* ── DEFECTIVE ─────────────────────────────────────── */}
                  {!isPass && (
                    <>
                      {/* Primary defect summary */}
                      {result.primary_defect && (
                        <div className="rounded-lg p-3 border"
                          style={{
                            borderColor: `${getClassColor(result.primary_defect).border}40`,
                            backgroundColor: getClassColor(result.primary_defect).bg
                          }}>
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center gap-2">
                              <Tag className="w-3.5 h-3.5" style={{ color: getClassColor(result.primary_defect).text }} />
                              <span className="text-xs font-semibold uppercase tracking-wider text-secondary">Primary Defect</span>
                            </div>
                          </div>
                          <p className="text-xl font-black capitalize" style={{ color: getClassColor(result.primary_defect).text }}>
                            {result.primary_defect}
                          </p>
                          {result.confidence != null && (
                            <ConfBar value={result.confidence} />
                          )}
                        </div>
                      )}

                      {/* ── YOLO detections with bounding boxes ─────── */}
                      {hasYoloDetections && (
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
                                <div key={i} className="rounded-lg p-3 border transition-all"
                                  style={{ borderColor: `${color.border}30`, backgroundColor: color.bg }}>
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

                      {/* ── Classifier-only result (no bbox) ────────── */}
                      {hasClassifierOnly && result.classification && (
                        <div>
                          <div className="flex items-center justify-between mb-2 px-0.5">
                            <p className="text-xs font-semibold text-secondary uppercase tracking-wider flex items-center gap-1.5">
                              <Cpu className="w-3.5 h-3.5" /> Classifier Result
                            </p>
                          </div>
                          {(() => {
                            const color = getClassColor(result.classification.class_name);
                            return (
                              <div className="rounded-lg p-3 border"
                                style={{ borderColor: `${color.border}30`, backgroundColor: color.bg }}>
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

                      {/* Summary table if multiple YOLO detections */}
                      {hasYoloDetections && result.detections.length > 1 && (
                        <div className="card overflow-hidden">
                          <table className="data-table">
                            <thead>
                              <tr>
                                <th>Defect Type</th>
                                <th>Confidence</th>
                              </tr>
                            </thead>
                            <tbody>
                              {result.detections.map((det, i) => (
                                <tr key={i}>
                                  <td className="font-medium capitalize text-dark">{det.class_name}</td>
                                  <td className="font-semibold text-danger">{(det.confidence * 100).toFixed(1)}%</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}

                      {/* Classifier class breakdown if available */}
                      {result.classification?.all_probs && (
                        <div className="card overflow-hidden">
                          <div className="px-3 py-2 bg-surface-2 border-b border-border">
                            <p className="text-xs font-semibold text-secondary uppercase tracking-wider flex items-center gap-1.5">
                              <Cpu className="w-3 h-3" /> Classifier Probabilities
                            </p>
                          </div>
                          <div className="p-3 space-y-2">
                            {Object.entries(result.classification.all_probs)
                              .sort(([, a], [, b]) => b - a)
                              .map(([cls, prob]) => (
                                <div key={cls} className="flex items-center gap-2 text-xs">
                                  <span className="w-24 shrink-0 capitalize text-secondary truncate">{cls}</span>
                                  <div className="flex-1 h-1.5 bg-border rounded-full overflow-hidden">
                                    <div className="h-full rounded-full"
                                      style={{
                                        width: `${(prob * 100).toFixed(1)}%`,
                                        backgroundColor: getClassColor(cls).border
                                      }} />
                                  </div>
                                  <span className="w-10 text-right font-semibold tabular-nums text-dark">
                                    {(prob * 100).toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-border bg-surface py-4">
        <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-muted">
          <p>Fabric Defect Inspection &mdash; AI-Powered Quality Control</p>
          <p>&bull; Fabric Defects Dataset (MobileNetV3) &bull; FastAPI + React</p>
        </div>
      </footer>
    </div>
  );
}
