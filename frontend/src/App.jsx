import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle, AlertTriangle, ShieldCheck, Activity, Image as ImageIcon, Info, Camera, Zap, Layers, RefreshCw } from 'lucide-react';
import clsx from 'clsx';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [activeTab, setActiveTab] = useState('inspection'); // 'inspection' or 'about'
  const imageRef = useRef(null);
  const containerRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    } else {
      setError("Please select a valid image file.");
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    } else {
      setError("Please drop a valid image file.");
    }
  };

  const resetImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const inspectFabric = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred during inspection. Ensure the backend is running.");
    } finally {
      setIsProcessing(false);
    }
  };

  const onImageLoad = (e) => {
    setImageSize({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight
    });
  };

  const renderBoundingBoxes = () => {
    if (!result || result.defects.length === 0 || !imageSize.width || !imageRef.current || !containerRef.current) return null;

    const displayedWidth = imageRef.current.clientWidth;
    const displayedHeight = imageRef.current.clientHeight;
    
    const scaleX = displayedWidth / imageSize.width;
    const scaleY = displayedHeight / imageSize.height;

    return result.defects.map((defect, index) => {
      const { x1, y1, x2, y2 } = defect.bbox;
      const left = x1 * scaleX;
      const top = y1 * scaleY;
      const width = (x2 - x1) * scaleX;
      const height = (y2 - y1) * scaleY;

      return (
        <div
          key={index}
          className="absolute border-2 border-danger z-10 transition-all duration-300"
          style={{
            left: `${left}px`,
            top: `${top}px`,
            width: `${width}px`,
            height: `${height}px`,
          }}
        >
          <div className="absolute -top-6 left-0 bg-danger text-white text-xs font-bold px-2 py-1 rounded-sm whitespace-nowrap">
            {defect.class_name.toUpperCase()} ({(defect.confidence * 100).toFixed(1)}%)
          </div>
        </div>
      );
    });
  };

  return (
    <div className="min-h-screen bg-background flex flex-col font-sans">
      {/* Navbar */}
      <nav className="border-b border-white/10 bg-surface/50 backdrop-blur-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="bg-primary/20 p-2 rounded-lg">
                <ShieldCheck className="text-primary w-6 h-6" />
              </div>
              <div>
                <h1 className="font-bold text-xl tracking-tight text-white">AI Fabric Inspection</h1>
                <p className="text-xs text-secondary font-medium tracking-wide">COMPUTER VISION QUALITY CONTROL</p>
              </div>
            </div>
            <div className="flex space-x-1 bg-surface/50 p-1 rounded-lg border border-white/5">
              <button 
                onClick={() => setActiveTab('inspection')}
                className={clsx(
                  "px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2",
                  activeTab === 'inspection' ? "bg-primary text-white shadow-md" : "text-secondary hover:text-white hover:bg-white/5"
                )}
              >
                <Camera className="w-4 h-4" />
                Dashboard
              </button>
              <button 
                onClick={() => setActiveTab('about')}
                className={clsx(
                  "px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2",
                  activeTab === 'about' ? "bg-primary text-white shadow-md" : "text-secondary hover:text-white hover:bg-white/5"
                )}
              >
                <Info className="w-4 h-4" />
                Project Info
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-grow max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
        
        {activeTab === 'inspection' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* Left Column - Upload & Preview */}
            <div className="lg:col-span-8 flex flex-col gap-6">
              
              <div className="glass-panel p-6 flex flex-col h-full min-h-[500px]">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <ImageIcon className="w-5 h-5 text-primary" />
                    Fabric Image
                  </h2>
                  {previewUrl && (
                    <button 
                      onClick={resetImage}
                      className="text-xs font-medium text-secondary hover:text-white flex items-center gap-1 transition-colors"
                    >
                      <RefreshCw className="w-3 h-3" />
                      Reset Image
                    </button>
                  )}
                </div>

                {!previewUrl ? (
                  <div 
                    className="flex-grow border-2 border-dashed border-white/20 rounded-xl flex flex-col items-center justify-center p-8 bg-black/20 hover:bg-black/30 transition-colors cursor-pointer group"
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    onClick={() => document.getElementById('file-upload').click()}
                  >
                    <div className="bg-surface p-4 rounded-full mb-4 group-hover:scale-110 transition-transform duration-300">
                      <UploadCloud className="w-8 h-8 text-primary" />
                    </div>
                    <p className="font-medium text-lg mb-2">Drag and drop fabric image here</p>
                    <p className="text-sm text-secondary mb-6">Supports JPG, PNG, JPEG</p>
                    <button className="bg-white/10 hover:bg-white/20 px-6 py-2 rounded-lg font-medium transition-colors">
                      Browse Files
                    </button>
                    <input 
                      id="file-upload" 
                      type="file" 
                      accept="image/*" 
                      className="hidden" 
                      onChange={handleFileSelect}
                    />
                  </div>
                ) : (
                  <div className="flex-grow flex flex-col items-center justify-center relative bg-black/40 rounded-xl overflow-hidden border border-white/10 p-4">
                    <div className="relative inline-block max-h-full max-w-full" ref={containerRef}>
                      <img 
                        ref={imageRef}
                        src={previewUrl} 
                        alt="Fabric Preview" 
                        className="max-h-[60vh] object-contain rounded-md shadow-2xl"
                        onLoad={onImageLoad}
                      />
                      {renderBoundingBoxes()}
                    </div>
                  </div>
                )}

                {previewUrl && (
                  <div className="mt-6 flex justify-end">
                    <button 
                      onClick={inspectFabric}
                      disabled={isProcessing}
                      className={clsx(
                        "flex items-center gap-2 px-8 py-3 rounded-lg font-bold text-white shadow-lg transition-all",
                        isProcessing 
                          ? "bg-primary/50 cursor-not-allowed" 
                          : "bg-primary hover:bg-blue-600 hover:shadow-primary/25 hover:-translate-y-0.5"
                      )}
                    >
                      {isProcessing ? (
                        <>
                          <RefreshCw className="w-5 h-5 animate-spin" />
                          Analyzing Fabric...
                        </>
                      ) : (
                        <>
                          <Activity className="w-5 h-5" />
                          Inspect Fabric
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - Results */}
            <div className="lg:col-span-4 flex flex-col gap-6">
              
              <div className="glass-panel p-6">
                <h2 className="text-lg font-semibold flex items-center gap-2 mb-6">
                  <Activity className="w-5 h-5 text-primary" />
                  Inspection Results
                </h2>

                {error && (
                  <div className="bg-danger/10 border border-danger/20 text-danger p-4 rounded-lg flex items-start gap-3 mb-6">
                    <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
                    <p className="text-sm leading-relaxed">{error}</p>
                  </div>
                )}

                {!result && !error && !isProcessing && (
                  <div className="text-center py-12 text-secondary flex flex-col items-center">
                    <Zap className="w-12 h-12 mb-4 opacity-20" />
                    <p>Upload an image and run inspection to see results here.</p>
                  </div>
                )}

                {isProcessing && (
                  <div className="space-y-4 py-8">
                    <div className="h-12 bg-white/5 rounded-lg animate-pulse"></div>
                    <div className="h-32 bg-white/5 rounded-lg animate-pulse"></div>
                    <div className="h-24 bg-white/5 rounded-lg animate-pulse"></div>
                  </div>
                )}

                {result && !isProcessing && (
                  <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    
                    {/* Quality Status Banner */}
                    <div className={clsx(
                      "p-4 rounded-xl flex items-center gap-4 mb-6 border shadow-lg",
                      result.quality_status === 'PASS' 
                        ? "bg-success/10 border-success/30 text-success" 
                        : "bg-danger/10 border-danger/30 text-danger"
                    )}>
                      {result.quality_status === 'PASS' ? (
                        <CheckCircle className="w-10 h-10 shrink-0" />
                      ) : (
                        <AlertTriangle className="w-10 h-10 shrink-0" />
                      )}
                      <div>
                        <p className="text-sm font-bold opacity-80 uppercase tracking-wider mb-0.5">Overall Quality</p>
                        <h3 className="text-2xl font-black uppercase tracking-tight">
                          {result.quality_status === 'PASS' ? 'PASSED' : 'DEFECTIVE'}
                        </h3>
                      </div>
                    </div>

                    {/* Defect Details */}
                    {result.quality_status === 'PASS' ? (
                      <div className="bg-white/5 border border-white/10 rounded-xl p-5 text-center">
                        <p className="text-white font-medium mb-1">No Defect Detected</p>
                        <p className="text-sm text-secondary">The fabric meets quality standards.</p>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="flex justify-between items-center px-1">
                          <h4 className="font-semibold text-sm text-secondary uppercase tracking-wider">Detected Defects</h4>
                          <span className="bg-danger/20 text-danger text-xs font-bold px-2.5 py-1 rounded-full">
                            {result.defect_count} Found
                          </span>
                        </div>
                        
                        <div className="space-y-3 max-h-[400px] overflow-y-auto pr-1 custom-scrollbar">
                          {result.defects.map((defect, idx) => (
                            <div key={idx} className="bg-black/30 border border-white/5 rounded-lg p-4 hover:border-danger/30 transition-colors">
                              <div className="flex justify-between items-start mb-3">
                                <div className="flex items-center gap-2">
                                  <div className="w-2 h-2 rounded-full bg-danger"></div>
                                  <span className="font-bold text-white capitalize text-lg">{defect.class_name}</span>
                                </div>
                                <div className="text-right">
                                  <span className="text-xs text-secondary block">Confidence</span>
                                  <span className="font-bold text-primary">{(defect.confidence * 100).toFixed(1)}%</span>
                                </div>
                              </div>
                              <div className="grid grid-cols-2 gap-2 text-xs text-secondary bg-white/5 rounded p-2">
                                <div><span className="opacity-50">X:</span> <span className="text-white/90">{defect.bbox.x1}</span></div>
                                <div><span className="opacity-50">Y:</span> <span className="text-white/90">{defect.bbox.y1}</span></div>
                                <div><span className="opacity-50">W:</span> <span className="text-white/90">{(defect.bbox.x2 - defect.bbox.x1).toFixed(1)}</span></div>
                                <div><span className="opacity-50">H:</span> <span className="text-white/90">{(defect.bbox.y2 - defect.bbox.y1).toFixed(1)}</span></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Stats Card */}
              <div className="glass-panel p-6">
                <h3 className="text-sm font-bold text-secondary uppercase tracking-wider mb-4 flex items-center gap-2">
                  <Layers className="w-4 h-4" />
                  Model Context
                </h3>
                <div className="grid grid-cols-2 gap-y-4 gap-x-2 text-sm">
                  <div>
                    <p className="text-secondary text-xs mb-1">Dataset</p>
                    <p className="font-medium text-white/90">TILDA-400</p>
                  </div>
                  <div>
                    <p className="text-secondary text-xs mb-1">Architecture</p>
                    <p className="font-medium text-white/90">YOLO Object Detection</p>
                  </div>
                  <div>
                    <p className="text-secondary text-xs mb-1">Classes</p>
                    <p className="font-medium text-white/90">4 Types</p>
                  </div>
                  <div>
                    <p className="text-secondary text-xs mb-1">Status</p>
                    <p className="font-medium text-success flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-success"></span> Active
                    </p>
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

        {activeTab === 'about' && (
          <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500 pb-12">
            <div className="glass-panel p-8 md:p-12">
              <h2 className="text-3xl font-black mb-6 tracking-tight">AI-Based Fabric Defect Detection</h2>
              
              <div className="prose prose-invert max-w-none">
                <div className="mb-8">
                  <h3 className="text-xl font-bold text-primary mb-3">Objective</h3>
                  <p className="text-secondary leading-relaxed">
                    Automate the process of fabric quality inspection using advanced Computer Vision techniques. Manual inspection is time-consuming, subjective, and prone to human error. This system provides consistent, real-time defect detection.
                  </p>
                </div>

                <div className="grid md:grid-cols-2 gap-8 mb-8">
                  <div className="bg-white/5 border border-white/10 p-6 rounded-xl">
                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                      <Layers className="w-5 h-5 text-primary" />
                      The Dataset
                    </h3>
                    <ul className="space-y-2 text-sm text-secondary">
                      <li><strong className="text-white/80">Name:</strong> TILDA-400</li>
                      <li><strong className="text-white/80">Task:</strong> Object Detection</li>
                      <li><strong className="text-white/80">Format:</strong> YOLO Annotations</li>
                      <li><strong className="text-white/80">Classes:</strong> Hole, Objects, Oil Spot, Thread Error</li>
                    </ul>
                  </div>

                  <div className="bg-white/5 border border-white/10 p-6 rounded-xl">
                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                      <Zap className="w-5 h-5 text-primary" />
                      Technology Stack
                    </h3>
                    <ul className="space-y-2 text-sm text-secondary">
                      <li><strong className="text-white/80">Core AI:</strong> Ultralytics YOLOv8</li>
                      <li><strong className="text-white/80">Backend:</strong> Python FastAPI</li>
                      <li><strong className="text-white/80">Frontend:</strong> React, TailwindCSS, Vite</li>
                      <li><strong className="text-white/80">Image Processing:</strong> OpenCV</li>
                    </ul>
                  </div>
                </div>

                <div className="mb-8">
                  <h3 className="text-xl font-bold text-primary mb-3">Future Scope</h3>
                  <ul className="list-disc pl-5 space-y-2 text-secondary leading-relaxed marker:text-primary">
                    <li>Real-time camera inspection on the production line</li>
                    <li>Defect severity estimation based on bounding box size</li>
                    <li>Automated quality reporting and dashboard analytics</li>
                    <li>Edge-device deployment for low-latency inference</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default App;
