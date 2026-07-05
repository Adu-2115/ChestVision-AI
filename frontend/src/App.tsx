import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'https://adu2115-chessvision-api.hf.space';

interface ModelScores {
  efficientnet_b0: number;
  mobilenet_v2: number;
  torchxrayvision: number | null;
}

interface Prediction {
  disease: string;
  probability: number;
  positive: boolean;
  model_scores?: ModelScores;
  disagreement?: number;
  n_models_used?: number;
}

interface DiseaseDetail {
  disease: string;
  probability: number;
  description: string;
  symptoms: string[];
  causes: string[];
  specialist: string;
  region: string;
}

interface Report {
  report_id: string;
  generated_at: string;
  patient_id: string;
  patient_age: number;
  patient_sex: string;
  image_filename: string;
  findings: string;
  differential: string;
  impression: string;
  recommendations: string[];
  disease_details: DiseaseDetail[];
  disclaimer: string;
  llm_generated: boolean;
  all_predictions: { disease: string; probability: number }[];
}

interface Result {
  scan_id: string;
  filename: string;
  age: number;
  sex: string;
  predictions: Prediction[];
  heatmaps: Record<string, string>;
  original: string;
  report: Report;
}

type TabType = 'predictions' | 'heatmaps' | 'report' | 'knowledge';

const MODEL_LABELS: Record<keyof ModelScores, string> = {
  efficientnet_b0: 'EfficientNet-B0',
  mobilenet_v2: 'MobileNetV2',
  torchxrayvision: 'TorchXRayVision',
};

export default function App() {
  const [file, setFile]                   = useState<File | null>(null);
  const [preview, setPreview]             = useState<string | null>(null);
  const [result, setResult]               = useState<Result | null>(null);
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState<string | null>(null);
  const [activeTab, setActiveTab]         = useState<TabType>('predictions');
  const [activeHeatmap, setActiveHeatmap] = useState<string | null>(null);
  const [patientAge, setPatientAge]       = useState<number>(60);
  const [patientSex, setPatientSex]       = useState<string>('Unknown');
  const [expandedDisease, setExpandedDisease] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const f = acceptedFiles[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setActiveHeatmap(null);
    setExpandedDisease(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png'],
      'application/dicom': ['.dcm'],
    },
    multiple: false,
  });

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('age',  patientAge.toString());
      formData.append('sex',  patientSex);

      const res = await axios.post(`${API_URL}/api/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      });
      setResult(res.data);
      setActiveTab('predictions');
      const firstHeatmap = Object.keys(res.data.heatmaps)[0];
      if (firstHeatmap) setActiveHeatmap(firstHeatmap);
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!result) return;
    try {
      const res = await axios.post(
        `${API_URL}/api/report/generate`,
        result.report,
        { responseType: 'blob' }
      );
      const url  = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href  = url;
      link.setAttribute('download', `ChestVision_${result.scan_id}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (e) {
      alert('PDF generation failed.');
    }
  };

  const tabs: { id: TabType; label: string; emoji: string }[] = [
    { id: 'predictions', label: 'Predictions', emoji: '📊' },
    { id: 'heatmaps',    label: 'Heatmaps',    emoji: '🔥' },
    { id: 'report',      label: 'Report',       emoji: '📋' },
    { id: 'knowledge',   label: 'Disease Info', emoji: '🧬' },
  ];

  const getConfidenceColor = (prob: number) => {
    if (prob >= 0.7) return 'text-red-400';
    if (prob >= 0.5) return 'text-orange-400';
    if (prob >= 0.3) return 'text-yellow-400';
    return 'text-gray-500';
  };

  const getBarColor = (prob: number) => {
    if (prob >= 0.7) return 'bg-red-500';
    if (prob >= 0.5) return 'bg-orange-500';
    if (prob >= 0.3) return 'bg-yellow-500';
    return 'bg-gray-600';
  };

  // Agreement signal: low disagreement = models concur, high = flag for review
  const getAgreementInfo = (disagreement?: number) => {
    if (disagreement === undefined) return null;
    if (disagreement < 0.15) return { label: 'Models agree', color: 'text-emerald-400', dot: 'bg-emerald-500' };
    if (disagreement < 0.35) return { label: 'Partial agreement', color: 'text-yellow-400', dot: 'bg-yellow-500' };
    return { label: 'Models disagree', color: 'text-red-400', dot: 'bg-red-500' };
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_#0f172a_0%,_#030712_60%)] text-gray-100 font-sans">

      {/* Header */}
      <header className="bg-gray-900/80 backdrop-blur border-b border-gray-800 px-6 py-4 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">🩻</span>
            <div>
              <h1 className="text-xl font-bold text-blue-400 leading-tight tracking-tight">ChestVision AI</h1>
              <p className="text-gray-500 text-xs">Explainable Chest X-Ray Disease Detection</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden sm:block relative group">
              <span className="flex items-center gap-1.5 text-xs bg-cyan-950 text-cyan-300 border border-cyan-800 px-3 py-1 rounded-full cursor-help">
                <span className="flex gap-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse [animation-delay:150ms]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse [animation-delay:300ms]" />
                </span>
                3-Model Ensemble
              </span>
              <div className="absolute right-0 top-full mt-2 w-56 bg-gray-800 border border-gray-700 rounded-lg p-3 text-xs text-gray-300 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-150 shadow-xl z-20">
                <p className="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1.5">Models in this ensemble</p>
                <ul className="space-y-1">
                  <li className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />EfficientNet-B0 (multimodal)</li>
                  <li className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />MobileNetV2 (multimodal)</li>
                  <li className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />TorchXRayVision (pretrained)</li>
                </ul>
              </div>
            </div>
            <span className="text-xs bg-yellow-950 text-yellow-400 border border-yellow-800 px-3 py-1 rounded-full">
              ⚠ Research Use Only
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left: Upload Panel */}
          <div className="lg:col-span-1 space-y-4">

            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500
                ${isDragActive ? 'border-blue-400 bg-blue-950 scale-105' : 'border-gray-700 hover:border-blue-600 hover:bg-gray-900 bg-gray-900'}`}
            >
              <input {...getInputProps()} />
              <div className="text-5xl mb-3">{isDragActive ? '📂' : '📁'}</div>
              <p className="text-gray-200 font-semibold text-sm">
                {isDragActive ? 'Drop X-Ray here...' : 'Upload Chest X-Ray'}
              </p>
              <p className="text-gray-500 text-xs mt-1">Drag & drop or click to browse</p>
              <p className="text-gray-700 text-xs mt-2">JPG · JPEG · PNG · DICOM</p>
            </div>

            {/* Preview */}
            {preview && (
              <div className="rounded-2xl overflow-hidden bg-gray-900 border border-gray-800 animate-[fadeIn_0.3s_ease-out]">
                <div className="relative bg-black">
                  <img src={preview} alt="X-Ray preview" className="w-full object-contain max-h-56 bg-black" />
                  {loading && (
                    <div className="absolute inset-0 overflow-hidden">
                      <div className="absolute left-0 right-0 h-8 bg-gradient-to-b from-transparent via-cyan-400/40 to-transparent animate-[scanSweep_1.8s_ease-in-out_infinite]" />
                    </div>
                  )}
                </div>
                <div className="px-4 py-2 flex items-center justify-between">
                  <p className="text-gray-400 text-xs truncate max-w-[70%]">{file?.name}</p>
                  <p className="text-gray-600 text-xs font-mono">{file ? (file.size / 1024).toFixed(0) + ' KB' : ''}</p>
                </div>
              </div>
            )}

            {/* Patient Info Form */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
              <p className="text-gray-400 text-xs font-semibold uppercase tracking-wider">
                Patient Information
              </p>
              <p className="text-gray-600 text-xs">
                Used by AI model and report generation for personalized analysis
              </p>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-gray-500 text-xs mb-1 block">Age</label>
                  <input
                    type="number"
                    min={0}
                    max={120}
                    value={patientAge}
                    onChange={e => setPatientAge(Number(e.target.value))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Age"
                  />
                </div>
                <div>
                  <label className="text-gray-500 text-xs mb-1 block">Sex</label>
                  <select
                    value={patientSex}
                    onChange={e => setPatientSex(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="Unknown">Unknown</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="w-full py-3 rounded-xl font-semibold text-sm transition-all duration-200
                bg-blue-600 hover:bg-blue-500 active:bg-blue-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-400
                disabled:bg-gray-800 disabled:text-gray-600 disabled:cursor-not-allowed text-white"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                  </svg>
                  Scanning with 3-model ensemble...
                </span>
              ) : <span>🔍 Analyze X-Ray</span>}
            </button>

            {/* Error */}
            {error && (
              <div className="bg-red-950 border border-red-800 rounded-xl p-4 animate-[fadeIn_0.2s_ease-out]">
                <p className="text-red-400 text-sm font-medium mb-0.5">Analysis failed</p>
                <p className="text-red-400/80 text-xs">{error}</p>
              </div>
            )}

            {/* Scan info */}
            {result && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-1 text-xs">
                <p className="text-gray-500 font-semibold uppercase tracking-wider mb-2">Scan Info</p>
                <p className="text-gray-400"><span className="text-gray-600">ID: </span><span className="font-mono">{result.scan_id.slice(0, 8)}...</span></p>
                <p className="text-gray-400"><span className="text-gray-600">File: </span>{result.filename}</p>
                <p className="text-gray-400"><span className="text-gray-600">Patient: </span>{result.sex}, Age {result.age}</p>
                <p className="text-gray-400"><span className="text-gray-600">Time: </span><span className="font-mono">{result.report.generated_at}</span></p>
                <p className="text-gray-400">
                  <span className="text-gray-600">Report: </span>
                  <span className={result.report.llm_generated ? 'text-green-400' : 'text-gray-500'}>
                    {result.report.llm_generated ? '🤖 LLaMA3-70B' : '📋 Template'}
                  </span>
                </p>
              </div>
            )}

            {/* Disclaimer */}
            <div className="bg-yellow-950 border border-yellow-900 rounded-xl p-4">
              <p className="text-yellow-500 text-xs font-semibold mb-1">⚠ Medical Disclaimer</p>
              <p className="text-yellow-300 text-xs leading-relaxed">
                This tool is for research and decision-support only.
                Not a substitute for professional medical diagnosis.
                All findings must be verified by a qualified clinician.
              </p>
            </div>
          </div>

          {/* Right: Results Panel */}
          <div className="lg:col-span-2">
            {!result ? (
              <div className="flex flex-col items-center justify-center min-h-96 bg-gray-900 rounded-2xl border border-gray-800 text-center p-8">
                <div className="relative w-20 h-20 mb-5 rounded-full border-2 border-dashed border-gray-700 flex items-center justify-center">
                  <span className="text-3xl opacity-40">🩻</span>
                </div>
                <p className="text-gray-500 text-lg font-medium">Awaiting scan</p>
                <p className="text-gray-700 text-sm mt-2 max-w-xs">Upload a chest X-ray and click Analyze — EfficientNet-B0, MobileNetV2, and TorchXRayVision will each independently assess it</p>
              </div>
            ) : (
              <div className="space-y-4">

                {/* Tabs */}
                <div className="flex gap-1 bg-gray-900 p-1 rounded-xl border border-gray-800">
                  {tabs.map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex-1 py-2 px-2 rounded-lg text-xs font-medium transition-all duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500
                        ${activeTab === tab.id ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-500 hover:text-gray-300'}`}
                    >
                      <span className="hidden sm:inline">{tab.emoji} </span>{tab.label}
                    </button>
                  ))}
                </div>

                {/* Tab: Predictions */}
                {activeTab === 'predictions' && (
                  <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 space-y-3">
                    <div className="flex items-center justify-between mb-2">
                      <h2 className="font-semibold text-gray-200">Disease Predictions</h2>
                      <span className="text-xs text-gray-600 font-mono flex items-center gap-1.5">
                        <span className="inline-block w-2 border-t border-dashed border-gray-500" />
                        threshold 50%
                      </span>
                    </div>
                    {result.predictions.map(p => {
                      const pct = Math.round(p.probability * 100);
                      const agreement = getAgreementInfo(p.disagreement);
                      const isExpanded = expandedDisease === p.disease;
                      const hasModelScores = !!p.model_scores;

                      return (
                        <div key={p.disease} className="rounded-xl border border-gray-800/60 overflow-hidden">
                          <button
                            onClick={() => hasModelScores && setExpandedDisease(isExpanded ? null : p.disease)}
                            className={`w-full text-left px-3 py-2.5 space-y-1.5 transition-colors ${hasModelScores ? 'hover:bg-gray-850 cursor-pointer' : 'cursor-default'}`}
                          >
                            <div className="flex justify-between items-center">
                              <div className="flex items-center gap-2">
                                <span className={`text-xs px-2 py-0.5 rounded-full font-medium
                                  ${p.positive ? 'bg-red-950 text-red-400 border border-red-900' : 'bg-gray-800 text-gray-500'}`}>
                                  {p.positive ? 'POSITIVE' : 'negative'}
                                </span>
                                <span className={`text-sm font-medium ${getConfidenceColor(p.probability)}`}>{p.disease}</span>
                                {agreement && (
                                  <span className={`hidden sm:flex items-center gap-1 text-[10px] ${agreement.color}`}>
                                    <span className={`w-1.5 h-1.5 rounded-full ${agreement.dot}`} />
                                    {agreement.label}
                                  </span>
                                )}
                              </div>
                              <div className="flex items-center gap-2">
                                <span className={`text-sm font-bold tabular-nums font-mono ${getConfidenceColor(p.probability)}`}>{pct}%</span>
                                {hasModelScores && (
                                  <span className={`text-gray-600 text-xs transition-transform ${isExpanded ? 'rotate-180' : ''}`}>▾</span>
                                )}
                              </div>
                            </div>
                            <div className="relative w-full bg-gray-800 rounded-full h-2">
                              <div className={`h-2 rounded-full transition-all duration-500 ${getBarColor(p.probability)}`} style={{ width: `${pct}%` }} />
                              <div className="absolute top-1/2 -translate-y-1/2 left-1/2 w-px h-3 bg-gray-500/70" title="50% threshold" />
                            </div>
                          </button>

                          {/* Expanded: per-model breakdown */}
                          {isExpanded && hasModelScores && (
                            <div className="px-3 pb-3 pt-1 bg-gray-950/60 border-t border-gray-800/60 space-y-2 animate-[fadeIn_0.15s_ease-out]">
                              {(Object.keys(MODEL_LABELS) as (keyof ModelScores)[]).map(key => {
                                const score = p.model_scores![key];
                                if (score === null || score === undefined) {
                                  return (
                                    <div key={key} className="flex items-center justify-between text-xs">
                                      <span className="text-gray-600">{MODEL_LABELS[key]}</span>
                                      <span className="text-gray-700 italic">not covered</span>
                                    </div>
                                  );
                                }
                                const modelPct = Math.round(score * 100);
                                return (
                                  <div key={key} className="flex items-center gap-2 text-xs">
                                    <span className="text-gray-500 w-32 shrink-0">{MODEL_LABELS[key]}</span>
                                    <div className="flex-1 bg-gray-800 rounded-full h-1.5">
                                      <div className="h-1.5 rounded-full bg-cyan-500/80" style={{ width: `${modelPct}%` }} />
                                    </div>
                                    <span className="text-gray-400 tabular-nums font-mono w-9 text-right">{modelPct}%</span>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      );
                    })}
                    <div className="mt-4 bg-gray-800 rounded-xl p-4 border border-gray-700">
                      <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Summary</p>
                      <p className="text-gray-300 text-sm leading-relaxed">{result.report.impression}</p>
                    </div>
                  </div>
                )}

                {/* Tab: Heatmaps */}
                {activeTab === 'heatmaps' && (
                  <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 space-y-4">
                    <h2 className="font-semibold text-gray-200">Grad-CAM Explainability</h2>
                    <p className="text-gray-600 text-xs">
                      Red/yellow regions indicate areas EfficientNet-B0 focused on for each prediction.
                    </p>
                    {Object.keys(result.heatmaps).length > 1 && (
                      <div className="flex gap-2 flex-wrap">
                        {Object.keys(result.heatmaps).map(disease => (
                          <button key={disease} onClick={() => setActiveHeatmap(disease)}
                            className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500
                              ${activeHeatmap === disease ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}>
                            {disease}
                          </button>
                        ))}
                      </div>
                    )}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <p className="text-xs text-gray-500 text-center">Original X-Ray</p>
                        <img src={`data:image/png;base64,${result.original}`} alt="Original" className="w-full rounded-xl object-contain bg-black" />
                      </div>
                      <div className="space-y-2">
                        <p className="text-xs text-blue-400 text-center">
                          {activeHeatmap || Object.keys(result.heatmaps)[0]} — Grad-CAM
                        </p>
                        <img
                          src={`data:image/png;base64,${result.heatmaps[activeHeatmap || Object.keys(result.heatmaps)[0]]}`}
                          alt="Heatmap" className="w-full rounded-xl object-contain bg-black"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* Tab: Report */}
                {activeTab === 'report' && (
                  <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 space-y-5">
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="font-semibold text-gray-200">Clinical Report</h2>
                        <p className="text-xs mt-0.5">
                          {result.report.llm_generated
                            ? <span className="text-green-400">🤖 AI-Generated (LLaMA3-70B)</span>
                            : <span className="text-gray-500">📋 Template-Based</span>}
                        </p>
                      </div>
                      <button onClick={handleDownloadReport}
                        className="text-xs bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-400">
                        ⬇ Download PDF
                      </button>
                    </div>

                    <div className="grid grid-cols-2 gap-3 bg-gray-800 rounded-xl p-4 text-xs">
                      <div><p className="text-gray-600">Report ID</p><p className="text-gray-300 font-mono">{result.report.report_id}</p></div>
                      <div><p className="text-gray-600">Generated</p><p className="text-gray-300 font-mono">{result.report.generated_at}</p></div>
                      <div><p className="text-gray-600">Patient</p><p className="text-gray-300">{result.sex}, Age {result.age}</p></div>
                      <div><p className="text-gray-600">Image</p><p className="text-gray-300 truncate">{result.filename}</p></div>
                    </div>

                    <div>
                      <p className="text-xs text-blue-400 uppercase tracking-wider font-semibold mb-2">Findings</p>
                      <p className="text-gray-300 text-sm leading-relaxed bg-gray-800 rounded-xl p-4">{result.report.findings}</p>
                    </div>

                    {result.report.differential && (
                      <div>
                        <p className="text-xs text-purple-400 uppercase tracking-wider font-semibold mb-2">Differential Diagnosis</p>
                        <p className="text-gray-300 text-sm leading-relaxed bg-gray-800 rounded-xl p-4 whitespace-pre-line">{result.report.differential}</p>
                      </div>
                    )}

                    <div>
                      <p className="text-xs text-blue-400 uppercase tracking-wider font-semibold mb-2">Impression</p>
                      <p className="text-gray-300 text-sm leading-relaxed bg-gray-800 rounded-xl p-4">{result.report.impression}</p>
                    </div>

                    <div>
                      <p className="text-xs text-blue-400 uppercase tracking-wider font-semibold mb-2">Recommendations</p>
                      <ul className="space-y-2">
                        {result.report.recommendations.map((r, i) => (
                          <li key={i} className="flex gap-2 text-sm text-gray-300 bg-gray-800 rounded-xl px-4 py-3">
                            <span className="text-blue-500 mt-0.5">•</span><span>{r}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="bg-yellow-950 border border-yellow-900 rounded-xl p-4">
                      <p className="text-yellow-300 text-xs leading-relaxed">{result.report.disclaimer}</p>
                    </div>
                  </div>
                )}

                {/* Tab: Disease Knowledge */}
                {activeTab === 'knowledge' && (
                  <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 space-y-4">
                    <h2 className="font-semibold text-gray-200">Disease Information</h2>
                    {result.report.disease_details.length === 0 ? (
                      <div className="text-center py-8">
                        <p className="text-gray-600 text-sm">No positive findings detected.</p>
                      </div>
                    ) : (
                      result.report.disease_details.map(d => (
                        <div key={d.disease} className="border border-gray-800 rounded-2xl p-5 space-y-3">
                          <div className="flex justify-between items-start">
                            <div>
                              <h3 className="font-bold text-blue-400 text-base">{d.disease}</h3>
                              <p className="text-xs text-gray-600 mt-0.5">Region: {d.region}</p>
                            </div>
                            <span className="text-red-400 font-bold text-lg">{Math.round(d.probability * 100)}%</span>
                          </div>
                          <p className="text-gray-300 text-sm leading-relaxed">{d.description}</p>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="bg-gray-800 rounded-xl p-3">
                              <p className="text-gray-500 text-xs font-semibold uppercase tracking-wider mb-2">Symptoms</p>
                              <ul className="space-y-1">
                                {d.symptoms.map(s => (
                                  <li key={s} className="text-gray-400 text-xs flex items-start gap-1">
                                    <span className="text-yellow-600 mt-0.5">•</span>{s}
                                  </li>
                                ))}
                              </ul>
                            </div>
                            <div className="bg-gray-800 rounded-xl p-3">
                              <p className="text-gray-500 text-xs font-semibold uppercase tracking-wider mb-2">Causes</p>
                              <ul className="space-y-1">
                                {d.causes.map(c => (
                                  <li key={c} className="text-gray-400 text-xs flex items-start gap-1">
                                    <span className="text-red-700 mt-0.5">•</span>{c}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 bg-green-950 border border-green-900 rounded-xl px-4 py-2">
                            <span className="text-green-500 text-xs">👨‍⚕️ Recommended Specialist:</span>
                            <span className="text-green-300 text-xs font-semibold">{d.specialist}</span>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}

              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
