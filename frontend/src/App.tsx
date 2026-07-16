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
  scan_db_id?: number | null;
  from_cache?: boolean;
}

type TabType = 'predictions' | 'heatmaps' | 'report' | 'knowledge';

const MODEL_LABELS: Record<keyof ModelScores, string> = {
  efficientnet_b0: 'EfficientNet-B0',
  mobilenet_v2: 'MobileNetV2',
  torchxrayvision: 'TorchXRayVision',
};

// Modern Inline SVGs for clinical look
const StethoscopeIcon = () => (
  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 14l9-5-9-5-9 5 9 5z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 14V22" />
  </svg>
);

const ActivityIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z" />
  </svg>
);

const HeatmapIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.879 16.121A3 3 0 1012.015 11L11 14.5c.373.16.713.407 1 .707z" />
  </svg>
);

const ReportIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

const WikiIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
);

const ShieldIcon = () => (
  <svg className="w-5 h-5 text-amber-500 mr-2 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
  </svg>
);

const UploadIcon = () => (
  <svg className="w-10 h-10 text-teal-500/70 mb-3 mx-auto transition-transform group-hover:scale-110" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
);

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
  const [feedbackChoice, setFeedbackChoice]   = useState<'correct' | 'incorrect' | null>(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackComments, setFeedbackComments]     = useState('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const f = acceptedFiles[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setActiveHeatmap(null);
    setExpandedDisease(null);
    setFeedbackChoice(null);
    setFeedbackSubmitted(false);
    setFeedbackComments('');
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png'],
      'application/dicom': ['.dcm'],
    },
    multiple: false,
  });

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setActiveHeatmap(null);
    setExpandedDisease(null);
    setActiveTab('predictions');
    setFeedbackChoice(null);
    setFeedbackSubmitted(false);
    setFeedbackComments('');
  };

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
        timeout: 300000,
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

  const handleSubmitFeedback = async (isCorrect: boolean) => {
    if (!result?.scan_db_id) {
      alert('Feedback unavailable for this scan (no database record).');
      return;
    }
    setFeedbackChoice(isCorrect ? 'correct' : 'incorrect');

    if (isCorrect) {
      setFeedbackLoading(true);
      try {
        await axios.post(`${API_URL}/api/feedback`, {
          scan_db_id: result.scan_db_id,
          is_correct: true,
        });
        setFeedbackSubmitted(true);
      } catch (e) {
        alert('Failed to submit feedback. Please try again.');
      } finally {
        setFeedbackLoading(false);
      }
    }
  };

  const handleSubmitCorrection = async () => {
    if (!result?.scan_db_id) return;
    setFeedbackLoading(true);
    try {
      await axios.post(`${API_URL}/api/feedback`, {
        scan_db_id: result.scan_db_id,
        is_correct: false,
        corrected_diagnosis: null,
        comments: feedbackComments || null,
      });
      setFeedbackSubmitted(true);
    } catch (e) {
      alert('Failed to submit feedback. Please try again.');
    } finally {
      setFeedbackLoading(false);
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

  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'predictions', label: 'Predictions', icon: <ActivityIcon /> },
    { id: 'heatmaps',    label: 'Heatmaps',    icon: <HeatmapIcon /> },
    { id: 'report',      label: 'Clinical Report', icon: <ReportIcon /> },
    { id: 'knowledge',   label: 'Disease Wiki', icon: <WikiIcon /> },
  ];

  const getConfidenceColor = (prob: number) => {
    if (prob >= 0.7) return 'text-rose-400';
    if (prob >= 0.5) return 'text-amber-500';
    if (prob >= 0.3) return 'text-yellow-400';
    return 'text-emerald-400';
  };

  const getBarColor = (prob: number) => {
    if (prob >= 0.7) return 'bg-rose-500';
    if (prob >= 0.5) return 'bg-amber-500';
    if (prob >= 0.3) return 'bg-yellow-500';
    return 'bg-emerald-500';
  };

  const getAgreementInfo = (disagreement?: number) => {
    if (disagreement === undefined) return null;
    if (disagreement < 0.15) return { label: 'Ensemble Concordant', color: 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20', dot: 'bg-emerald-400' };
    if (disagreement < 0.35) return { label: 'Partial Agreement', color: 'text-yellow-400 bg-yellow-500/10 border border-yellow-500/20', dot: 'bg-yellow-400' };
    return { label: 'Ensemble Discordant (Flagged)', color: 'text-rose-400 bg-rose-500/10 border border-rose-500/20', dot: 'bg-rose-400' };
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 medical-grid relative overflow-x-hidden">
      
      {/* Decorative background glow */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-teal-500/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-10 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-[120px] pointer-events-none" />

      {/* Header */}
      <header className="bg-slate-900/60 backdrop-blur-md border-b border-slate-800/80 px-6 py-4 sticky top-0 z-20">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-teal-500/10 border border-teal-500/30 rounded-xl shadow-inner text-teal-400 animate-pulse-glow">
              <StethoscopeIcon />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-teal-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent">
                  ChestVision AI
                </h1>
                <span className="text-[10px] uppercase font-bold tracking-widest bg-teal-500/10 border border-teal-500/30 text-teal-400 px-2 py-0.5 rounded">
                  v2.0 Console
                </span>
              </div>
              <p className="text-slate-400 text-xs mt-0.5">Clinical Decision Support & Chest X-Ray Analytics</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden md:block relative group">
              <span className="flex items-center gap-2 text-xs bg-slate-800/80 text-cyan-300 border border-cyan-800/40 px-3.5 py-1.5 rounded-full cursor-help hover:bg-slate-800 transition-colors">
                <span className="flex gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse [animation-delay:200ms]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse [animation-delay:400ms]" />
                </span>
                3-Model Ensemble Active
              </span>
              <div className="absolute right-0 top-full mt-2.5 w-64 bg-slate-900 border border-slate-800 rounded-xl p-3.5 text-xs text-slate-300 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 shadow-2xl z-30">
                <p className="text-slate-400 uppercase tracking-widest text-[9px] font-bold mb-2">Ensemble Configurations</p>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">EfficientNet-B0</span>
                    <span className="text-teal-400 font-mono text-[10px]">Multimodal</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">MobileNetV2</span>
                    <span className="text-teal-400 font-mono text-[10px]">Multimodal</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">TorchXRayVision</span>
                    <span className="text-teal-400 font-mono text-[10px]">Pretrained</span>
                  </div>
                </div>
              </div>
            </div>
            <span className="text-xs bg-amber-500/10 text-amber-400 border border-amber-500/30 px-3.5 py-1.5 rounded-full font-medium">
              ⚠ Research & Clinical Evaluation Use Only
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

          {/* Left Column: Control Center (4 cols) */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800/80 rounded-2xl p-6 shadow-xl space-y-6">
              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wider text-teal-400 mb-1">Diagnostic Console</h3>
                <p className="text-slate-400 text-xs">Configure patient details and upload radiological images.</p>
              </div>

              {/* Dropzone */}
              <div
                {...getRootProps()}
                className={`group border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500
                  ${isDragActive 
                    ? 'border-teal-400 bg-teal-950/20 scale-[1.02]' 
                    : 'border-slate-800 hover:border-teal-500/50 hover:bg-slate-900/30 bg-slate-950/50'}`}
              >
                <input {...getInputProps()} />
                <UploadIcon />
                <p className="text-slate-200 font-semibold text-sm">
                  {isDragActive ? 'Release to drop scan...' : 'Upload Chest Radiograph'}
                </p>
                <p className="text-slate-500 text-xs mt-1">Drag & drop or click to browse directories</p>
                <div className="mt-3.5 flex items-center justify-center gap-1.5 text-[10px] text-slate-400 bg-slate-900/60 py-1 px-3 rounded-full w-max mx-auto border border-slate-800">
                  <span>DICOM</span>
                  <span className="text-slate-600">•</span>
                  <span>PNG</span>
                  <span className="text-slate-600">•</span>
                  <span>JPEG</span>
                </div>
              </div>

              {/* Preview with Holographic Laser Scanning Effect */}
              {preview && (
                <div className="rounded-xl overflow-hidden bg-slate-950 border border-slate-800/80 p-1.5 shadow-inner relative">
                  <div className="relative rounded-lg overflow-hidden bg-black aspect-square flex items-center justify-center">
                    <img src={preview} alt="X-Ray preview" className="w-full h-full object-contain" />
                    {loading && (
                      <div className="absolute inset-0 bg-teal-500/5">
                        <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-teal-400 to-transparent shadow-[0_0_10px_rgba(20,184,166,0.8)] animate-scan-sweep" />
                      </div>
                    )}
                  </div>
                  <div className="px-3 py-2 flex items-center justify-between text-xs text-slate-400">
                    <span className="truncate max-w-[70%] font-mono">{file?.name}</span>
                    <span className="text-slate-500 font-mono">{file ? (file.size / 1024).toFixed(0) + ' KB' : ''}</span>
                  </div>
                </div>
              )}

              {/* Patient Profile */}
              <div className="bg-slate-950/50 border border-slate-850 rounded-xl p-4.5 space-y-4">
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-3 bg-teal-500 rounded" />
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    Patient Metadata
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-slate-500 text-[10px] uppercase font-bold tracking-wider mb-1.5 block">Age (Years)</label>
                    <input
                      type="number"
                      min={0}
                      max={120}
                      value={patientAge}
                      onChange={e => setPatientAge(Number(e.target.value))}
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-teal-500/50 focus:border-teal-500 transition-colors font-mono"
                      placeholder="e.g. 60"
                    />
                  </div>
                  <div>
                    <label className="text-slate-500 text-[10px] uppercase font-bold tracking-wider mb-1.5 block">Biological Sex</label>
                    <select
                      value={patientSex}
                      onChange={e => setPatientSex(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-teal-500/50 focus:border-teal-500 transition-colors"
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
                className="w-full relative py-3 px-4 rounded-xl font-semibold text-sm transition-all duration-300 overflow-hidden group/btn
                  bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 active:scale-[0.98]
                  disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-slate-950 font-bold shadow-lg shadow-teal-500/10 hover:shadow-teal-500/20"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2.5">
                    <svg className="animate-spin h-4 w-4 text-slate-950" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                    </svg>
                    Running Ensemble Diagnosis...
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-1.5">
                    <svg className="w-4 h-4 text-slate-950 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    Begin Diagnostic Scan
                  </span>
                )}
              </button>

              {/* Error Notification */}
              {error && (
                <div className="bg-rose-950/20 border border-rose-900/50 rounded-xl p-4 flex gap-3 items-start animate-[fadeIn_0.3s_ease-out]">
                  <span className="text-rose-400 text-lg">⚠</span>
                  <div>
                    <p className="text-rose-400 text-xs font-semibold mb-0.5">System Exception</p>
                    <p className="text-rose-300/80 text-[11px] leading-relaxed">{error}</p>
                  </div>
                </div>
              )}

              {/* Medical Disclaimer */}
              <div className="bg-slate-950/70 border border-slate-850 rounded-xl p-4.5 flex items-start">
                <ShieldIcon />
                <div className="space-y-1">
                  <p className="text-slate-300 text-[10px] uppercase font-bold tracking-wider">Clinical Guidance Directive</p>
                  <p className="text-slate-400 text-[11px] leading-relaxed">
                    This module operates exclusively as a clinical decision-support resource. Under no circumstances should results be interpreted as self-contained diagnostic findings. All outputs require verification by a credentialed clinician.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Visualization & Reports (8 cols) */}
          <div className="lg:col-span-8 space-y-6">
            {!result ? (
              <div className="flex flex-col items-center justify-center min-h-[500px] bg-slate-900/20 border border-slate-800/60 rounded-2xl text-center p-8 relative overflow-hidden">
                <div className="absolute inset-0 bg-slate-950/20 backdrop-blur-[2px]" />
                <div className="relative z-10 space-y-5">
                  <div className="w-16 h-16 mx-auto rounded-full bg-teal-500/5 border border-teal-500/20 flex items-center justify-center text-teal-400 animate-pulse-glow">
                    <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-slate-300 font-semibold text-base">Radiograph Awaiting Analysis</p>
                    <p className="text-slate-500 text-xs mt-2 max-w-md mx-auto leading-relaxed">
                      Upload a front-facing chest radiograph (DICOM or standard image formats) to activate predictions across MobileNetV2, EfficientNet-B0, and TorchXRayVision.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-6">

                {/* Dashboard Tabs & Actions */}
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 bg-slate-900/60 backdrop-blur-md border border-slate-800/80 p-2 rounded-2xl">
                  <div className="flex flex-wrap gap-1">
                    {tabs.map(tab => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex items-center gap-2 py-2.5 px-4 rounded-xl text-xs font-semibold tracking-wide transition-all duration-200 focus-visible:outline-none
                          ${activeTab === tab.id 
                            ? 'bg-teal-500 text-slate-950 shadow-md font-bold shadow-teal-500/10' 
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'}`}
                      >
                        {tab.icon}
                        <span>{tab.label}</span>
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={handleReset}
                    className="flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-xl text-xs font-bold bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-slate-100 transition-all focus-visible:outline-none"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 7.89H18" />
                    </svg>
                    New Assessment
                  </button>
                </div>

                {/* Active Scan Metadata Header Card */}
                <div className="bg-slate-900/40 border border-slate-805 rounded-2xl p-5 flex flex-wrap items-center justify-between gap-4">
                  <div className="flex flex-wrap items-center gap-5 text-xs text-slate-400">
                    <div>
                      <span className="text-slate-500 uppercase tracking-widest text-[9px] block">Diagnosis ID</span>
                      <span className="font-mono text-slate-200 font-semibold">{result.scan_id.slice(0, 16)}...</span>
                    </div>
                    <div className="w-px h-6 bg-slate-800 hidden sm:block" />
                    <div>
                      <span className="text-slate-500 uppercase tracking-widest text-[9px] block">Demographics</span>
                      <span className="text-slate-200 font-semibold">{result.sex}, Age {result.age}</span>
                    </div>
                    <div className="w-px h-6 bg-slate-800 hidden sm:block" />
                    <div>
                      <span className="text-slate-500 uppercase tracking-widest text-[9px] block">Image Filename</span>
                      <span className="text-slate-200 font-semibold truncate max-w-[150px] inline-block align-bottom">{result.filename}</span>
                    </div>
                  </div>
                  {result.from_cache && (
                    <span className="text-[10px] font-bold tracking-wider bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 px-2.5 py-1 rounded-full uppercase">
                      Cached Diagnostic File
                    </span>
                  )}
                </div>

                {/* Feedback Panel */}
                {result.scan_db_id && (
                  <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 shadow-sm">
                    {feedbackSubmitted ? (
                      <div className="flex items-center gap-2.5 text-xs text-emerald-400">
                        <span className="flex items-center justify-center w-5 h-5 rounded-full bg-emerald-500/10 border border-emerald-500/30">✓</span>
                        <p className="font-medium">Diagnostic feedback logged successfully. Thank you for contributing to clinical refinement.</p>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between flex-wrap gap-4">
                          <div className="space-y-0.5">
                            <h4 className="text-xs font-semibold text-slate-200">Refine Model Performance</h4>
                            <p className="text-[11px] text-slate-400">Did this analysis assist in diagnostic review?</p>
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={() => handleSubmitFeedback(true)}
                              disabled={feedbackLoading}
                              className={`text-xs px-3.5 py-2 rounded-xl font-bold transition-all border
                                ${feedbackChoice === 'correct'
                                  ? 'bg-emerald-950 border-emerald-700 text-emerald-400 shadow-inner'
                                  : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-emerald-600/50 hover:text-emerald-400'}`}
                            >
                              👍 Yes, Helpful
                            </button>
                            <button
                              onClick={() => handleSubmitFeedback(false)}
                              disabled={feedbackLoading}
                              className={`text-xs px-3.5 py-2 rounded-xl font-bold transition-all border
                                ${feedbackChoice === 'incorrect'
                                  ? 'bg-rose-950 border-rose-800 text-rose-400 shadow-inner'
                                  : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-rose-600/50 hover:text-rose-400'}`}
                            >
                              👎 Discrepancy
                            </button>
                          </div>
                        </div>

                        {feedbackChoice === 'incorrect' && (
                          <div className="space-y-2.5 pt-1.5 border-t border-slate-800/40 animate-[fadeIn_0.2s_ease-out]">
                            <textarea
                              value={feedbackComments}
                              onChange={e => setFeedbackComments(e.target.value)}
                              placeholder="Describe findings discrepancy or submit clinical correction details..."
                              rows={2}
                              className="w-full bg-slate-950 border border-slate-850 rounded-xl px-3.5 py-2.5 text-slate-200 text-xs focus:outline-none focus:ring-1 focus:ring-teal-500/50 resize-none placeholder-slate-600"
                            />
                            <button
                              onClick={handleSubmitCorrection}
                              disabled={feedbackLoading}
                              className="text-xs bg-teal-500 hover:bg-teal-400 disabled:bg-slate-800 disabled:text-slate-500 text-slate-950 px-4 py-2 rounded-lg font-bold transition-colors"
                            >
                              {feedbackLoading ? 'Logging...' : 'Submit Clinical Feedback'}
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Tab Content Sections */}

                {/* TAB: Predictions */}
                {activeTab === 'predictions' && (
                  <div className="bg-slate-905 border border-slate-800 rounded-2xl p-6 space-y-6">
                    <div className="flex items-center justify-between pb-3 border-b border-slate-800/60">
                      <div>
                        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Ensemble Findings Matrix</h2>
                        <p className="text-slate-400 text-xs mt-0.5">Confidence indices aggregated across multiple models.</p>
                      </div>
                      <div className="flex items-center gap-2 text-[10px] text-slate-500 font-mono">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-600" />
                        <span>Aggregated Threshold: 50%</span>
                      </div>
                    </div>

                    <div className="space-y-4">
                      {result.predictions.map(p => {
                        const pct = Math.round(p.probability * 100);
                        const agreement = getAgreementInfo(p.disagreement);
                        const isExpanded = expandedDisease === p.disease;
                        const hasModelScores = !!p.model_scores;

                        return (
                          <div key={p.disease} className="rounded-xl border border-slate-800 bg-slate-950/20 overflow-hidden transition-all duration-200 hover:border-slate-700/80">
                            <button
                              onClick={() => hasModelScores && setExpandedDisease(isExpanded ? null : p.disease)}
                              className={`w-full text-left px-4 py-3.5 space-y-3 transition-colors ${hasModelScores ? 'hover:bg-slate-900/30 cursor-pointer' : 'cursor-default'}`}
                            >
                              <div className="flex justify-between items-center">
                                <div className="flex items-center gap-3">
                                  <span className={`text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border
                                    ${p.positive 
                                      ? 'bg-rose-500/10 text-rose-400 border-rose-500/30 shadow-[0_0_5px_rgba(244,63,94,0.1)]' 
                                      : 'bg-slate-900 text-slate-500 border-slate-800'}`}>
                                    {p.positive ? 'Detected' : 'Normal'}
                                  </span>
                                  <span className="text-xs font-semibold text-slate-200">{p.disease}</span>
                                  {agreement && (
                                    <span className={`hidden sm:flex items-center gap-1.5 text-[10px] px-2 py-0.5 rounded-full ${agreement.color}`}>
                                      <span className={`w-1.5 h-1.5 rounded-full ${agreement.dot}`} />
                                      {agreement.label}
                                    </span>
                                  )}
                                </div>
                                <div className="flex items-center gap-3">
                                  <span className={`text-sm font-bold font-mono ${getConfidenceColor(p.probability)}`}>{pct}%</span>
                                  {hasModelScores && (
                                    <svg className={`w-4 h-4 text-slate-500 transition-transform duration-200 ${isExpanded ? 'rotate-180 text-teal-400' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                                      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                                    </svg>
                                  )}
                                </div>
                              </div>
                              <div className="relative w-full bg-slate-900 rounded-full h-2">
                                <div className={`h-2 rounded-full transition-all duration-700 ease-out ${getBarColor(p.probability)}`} style={{ width: `${pct}%` }} />
                                <div className="absolute top-1/2 -translate-y-1/2 left-1/2 w-px h-3 bg-slate-800" title="50% Threshold" />
                              </div>
                            </button>

                            {/* Expanded sub-model details */}
                            {isExpanded && hasModelScores && (
                              <div className="px-4 pb-4 pt-1.5 bg-slate-950 border-t border-slate-850 space-y-3 animate-[fadeIn_0.2s_ease-out]">
                                <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Individual Model Probability Matrices</p>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                  {(Object.keys(MODEL_LABELS) as (keyof ModelScores)[]).map(key => {
                                    const score = p.model_scores![key];
                                    if (score === null || score === undefined) {
                                      return (
                                        <div key={key} className="bg-slate-900/60 p-3 rounded-lg border border-slate-850 flex items-center justify-between text-xs">
                                          <span className="text-slate-500 font-medium">{MODEL_LABELS[key]}</span>
                                          <span className="text-slate-600 italic">Not Cover</span>
                                        </div>
                                      );
                                    }
                                    const modelPct = Math.round(score * 100);
                                    return (
                                      <div key={key} className="bg-slate-900/60 p-3 rounded-lg border border-slate-850 space-y-2">
                                        <div className="flex justify-between text-xs">
                                          <span className="text-slate-400 font-medium">{MODEL_LABELS[key]}</span>
                                          <span className="text-slate-200 font-mono font-bold">{modelPct}%</span>
                                        </div>
                                        <div className="w-full bg-slate-950 rounded-full h-1.5">
                                          <div className="h-1.5 rounded-full bg-cyan-500/80" style={{ width: `${modelPct}%` }} />
                                        </div>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    <div className="bg-slate-950 border border-slate-850 rounded-xl p-5 space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                        <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">Automated Diagnosis Summary</span>
                      </div>
                      <p className="text-slate-300 text-xs leading-relaxed">{result.report.impression}</p>
                    </div>
                  </div>
                )}

                {/* TAB: Heatmaps */}
                {activeTab === 'heatmaps' && (
                  <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 space-y-5">
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                      <div>
                        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Grad-CAM Spatial Explainability</h2>
                        <p className="text-slate-400 text-xs mt-0.5">Saliency highlights indicating model regions-of-interest.</p>
                      </div>
                      {Object.keys(result.heatmaps).length > 1 && (
                        <div className="flex gap-1.5 flex-wrap bg-slate-950 p-1 rounded-xl border border-slate-850">
                          {Object.keys(result.heatmaps).map(disease => (
                            <button
                              key={disease}
                              onClick={() => setActiveHeatmap(disease)}
                              className={`text-[11px] px-3 py-1.5 rounded-lg font-semibold transition-all focus-visible:outline-none
                                ${activeHeatmap === disease 
                                  ? 'bg-teal-500 text-slate-950 font-bold' 
                                  : 'text-slate-400 hover:text-slate-200'}`}
                            >
                              {disease}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                      <div className="space-y-2 bg-slate-950 p-2.5 rounded-2xl border border-slate-850">
                        <p className="text-[10px] text-slate-500 text-center uppercase tracking-wider font-bold">Standard Radiograph</p>
                        <div className="rounded-lg overflow-hidden bg-black aspect-square flex items-center justify-center">
                          <img src={result.original ? `data:image/png;base64,${result.original}` : (preview || '')} alt="Original" className="w-full h-full object-contain" />
                        </div>
                      </div>
                      <div className="space-y-2 bg-slate-950 p-2.5 rounded-2xl border border-teal-500/20 shadow-[0_0_15px_rgba(20,184,166,0.05)]">
                        <p className="text-[10px] text-teal-400 text-center uppercase tracking-wider font-bold">
                          {Object.keys(result.heatmaps).length > 0 ? (activeHeatmap || Object.keys(result.heatmaps)[0]) : 'Grad-CAM'} Activations
                        </p>
                        <div className="rounded-lg overflow-hidden bg-black aspect-square flex items-center justify-center relative">
                          {Object.keys(result.heatmaps).length > 0 ? (
                            <img
                              src={`data:image/png;base64,${result.heatmaps[activeHeatmap || Object.keys(result.heatmaps)[0]]}`}
                              alt="Heatmap" className="w-full h-full object-contain"
                            />
                          ) : (
                            <div className="flex flex-col items-center justify-center p-6 text-center h-full min-h-[250px] bg-slate-900/40 rounded-lg">
                              <svg className="w-8 h-8 text-slate-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                              </svg>
                              <p className="text-slate-400 text-xs font-semibold">Heatmap Not Available</p>
                              <p className="text-slate-500 text-[10px] mt-1 max-w-[200px] leading-relaxed">
                                Explainability heatmaps are only generated on initial scans and are not stored in cache to optimize storage resources.
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB: Report */}
                {activeTab === 'report' && (
                  <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 space-y-6">
                    <div className="flex justify-between items-center flex-wrap gap-4 border-b border-slate-800/60 pb-4">
                      <div>
                        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Clinical Pathology Report</h2>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Format: {result.report.llm_generated
                            ? <span className="text-teal-400 font-bold">Structured AI (LLaMA-3)</span>
                            : <span className="text-slate-500 font-medium">Standardized Template</span>}
                        </p>
                      </div>
                      <button
                        onClick={handleDownloadReport}
                        className="text-xs bg-teal-500 hover:bg-teal-400 text-slate-950 px-4 py-2.5 rounded-xl font-bold transition-all flex items-center shadow-md shadow-teal-500/10"
                      >
                        <svg className="w-3.5 h-3.5 mr-1.5 text-slate-950" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Download PDF Report
                      </button>
                    </div>

                    {/* Official Medical Diagnostic Styling Sheet */}
                    <div className="bg-slate-950 border border-slate-850 rounded-2xl p-6 font-mono text-xs text-slate-300 space-y-6 shadow-inner relative overflow-hidden">
                      <div className="absolute right-6 top-6 opacity-[0.03] text-teal-400 pointer-events-none">
                        <svg className="w-32 h-32" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                        </svg>
                      </div>

                      {/* Header */}
                      <div className="flex justify-between items-start border-b border-slate-800/80 pb-4">
                        <div className="space-y-1">
                          <p className="text-teal-400 font-bold uppercase tracking-widest text-[10px]">ChestVision Clinic Console</p>
                          <p className="text-[10px] text-slate-500">Radiological Lab Reporting Service</p>
                        </div>
                        <div className="text-right">
                          <p className="text-slate-400 font-semibold">REF: {result.report.report_id.slice(0, 12)}</p>
                          <p className="text-[10px] text-slate-500">DATE: {result.report.generated_at}</p>
                        </div>
                      </div>

                      {/* Demographic Table */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 bg-slate-900/60 p-4 rounded-xl border border-slate-850">
                        <div>
                          <span className="text-slate-500 block uppercase text-[9px] tracking-wider mb-1">Patient Sex</span>
                          <span className="text-slate-200 font-semibold">{result.sex}</span>
                        </div>
                        <div>
                          <span className="text-slate-500 block uppercase text-[9px] tracking-wider mb-1">Patient Age</span>
                          <span className="text-slate-200 font-semibold">{result.age} Yrs</span>
                        </div>
                        <div>
                          <span className="text-slate-500 block uppercase text-[9px] tracking-wider mb-1">Modality</span>
                          <span className="text-slate-200 font-semibold">CR (Chest Chest)</span>
                        </div>
                        <div>
                          <span className="text-slate-500 block uppercase text-[9px] tracking-wider mb-1">Image Status</span>
                          <span className="text-emerald-400 font-semibold">Process Valid</span>
                        </div>
                      </div>

                      {/* Findings */}
                      <div className="space-y-2">
                        <h4 className="text-teal-400 font-bold uppercase tracking-wider text-[10px]">Clinical Findings</h4>
                        <p className="text-slate-350 leading-relaxed bg-slate-900/40 p-4 rounded-xl border border-slate-850 font-sans text-xs">{result.report.findings}</p>
                      </div>

                      {/* Differential */}
                      {result.report.differential && (
                        <div className="space-y-2">
                          <h4 className="text-amber-500 font-bold uppercase tracking-wider text-[10px]">Differential Diagnosis</h4>
                          <p className="text-slate-350 leading-relaxed bg-slate-900/40 p-4 rounded-xl border border-slate-850 font-sans text-xs whitespace-pre-line">{result.report.differential}</p>
                        </div>
                      )}

                      {/* Impression */}
                      <div className="space-y-2">
                        <h4 className="text-teal-400 font-bold uppercase tracking-wider text-[10px]">Impression</h4>
                        <p className="text-slate-350 leading-relaxed bg-slate-900/40 p-4 rounded-xl border border-slate-850 font-sans text-xs">{result.report.impression}</p>
                      </div>

                      {/* Recommendations */}
                      <div className="space-y-2">
                        <h4 className="text-teal-400 font-bold uppercase tracking-wider text-[10px]">Diagnostic Recommendations</h4>
                        <div className="bg-slate-900/40 border border-slate-850 rounded-xl p-4 font-sans space-y-2">
                          {result.report.recommendations.map((r, i) => (
                            <div key={i} className="flex gap-2 text-slate-350 text-xs">
                              <span className="text-teal-400 font-bold">•</span>
                              <span>{r}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Signature / Verification Area */}
                      <div className="border-t border-slate-800/80 pt-5 flex justify-between items-end flex-wrap gap-4 font-sans text-slate-400">
                        <div className="space-y-1">
                          <p className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">System Cryptographic Signature</p>
                          <p className="font-mono text-[9px] text-slate-400 truncate max-w-[280px] bg-slate-900 py-1 px-2.5 rounded border border-slate-850">CVAI_SECURE_HASH::{result.scan_id}</p>
                        </div>
                        <div className="text-right space-y-1 min-w-[120px]">
                          <div className="w-24 border-b border-slate-700 mx-auto md:mr-0 mb-1" />
                          <p className="text-[10px] text-slate-300 font-semibold">Radiology Sign-off</p>
                          <p className="text-[9px] text-slate-500">Authorized Clinician Required</p>
                        </div>
                      </div>

                      <div className="bg-slate-900/80 border border-slate-850 rounded-xl p-4 font-sans text-[11px] text-slate-500 leading-normal">
                        {result.report.disclaimer}
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB: Disease Knowledge Wiki */}
                {activeTab === 'knowledge' && (
                  <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 space-y-6">
                    <div>
                      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Pathology Wiki Reference</h2>
                      <p className="text-slate-400 text-xs mt-0.5">Clinical details on diagnosed anomalies.</p>
                    </div>

                    {result.report.disease_details.length === 0 ? (
                      <div className="text-center py-10 bg-slate-950/40 border border-slate-850 rounded-2xl">
                        <p className="text-slate-500 text-xs">No active chest pathologies were identified above critical threshold limits.</p>
                      </div>
                    ) : (
                      <div className="space-y-6">
                        {result.report.disease_details.map(d => (
                          <div key={d.disease} className="border border-slate-850 bg-slate-950/20 rounded-2xl p-5 space-y-4 hover:border-slate-800 transition-colors">
                            <div className="flex justify-between items-start flex-wrap gap-2">
                              <div>
                                <h3 className="font-bold text-teal-400 text-base">{d.disease}</h3>
                                <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-1.5 flex items-center gap-1.5">
                                  <span>Target Region: {d.region}</span>
                                </div>
                              </div>
                              <span className="text-rose-400 font-mono font-bold bg-rose-500/10 border border-rose-500/20 px-3 py-1 rounded-lg text-sm">{Math.round(d.probability * 100)}% Probability</span>
                            </div>
                            <p className="text-slate-300 text-xs leading-relaxed">{d.description}</p>
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="bg-slate-950 border border-slate-850 rounded-xl p-4.5 space-y-2">
                                <p className="text-slate-400 text-[10px] uppercase font-bold tracking-wider pb-1 border-b border-slate-900">Recognized Clinical Symptoms</p>
                                <ul className="space-y-1.5 pt-1">
                                  {d.symptoms.map(s => (
                                    <li key={s} className="text-slate-300 text-xs flex items-start gap-2">
                                      <span className="text-amber-500 mt-1.5 w-1 h-1 rounded-full shrink-0 bg-amber-500" />
                                      <span>{s}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                              <div className="bg-slate-950 border border-slate-850 rounded-xl p-4.5 space-y-2">
                                <p className="text-slate-400 text-[10px] uppercase font-bold tracking-wider pb-1 border-b border-slate-900">Etiology & Common Causes</p>
                                <ul className="space-y-1.5 pt-1">
                                  {d.causes.map(c => (
                                    <li key={c} className="text-slate-300 text-xs flex items-start gap-2">
                                      <span className="text-rose-500 mt-1.5 w-1 h-1 rounded-full shrink-0 bg-rose-500" />
                                      <span>{c}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>

                            <div className="flex items-center gap-2.5 bg-emerald-500/5 border border-emerald-500/20 rounded-xl px-4 py-3 text-xs">
                              <span className="text-emerald-400 font-bold shrink-0">👨‍⚕️ Clinical Specialist:</span>
                              <span className="text-slate-200 font-semibold">{d.specialist}</span>
                            </div>
                          </div>
                        ))}
                      </div>
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
