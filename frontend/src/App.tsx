import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'https://adu2115-chessvision-api.hf.space';

interface Prediction {
  disease: string;
  probability: number;
  positive: boolean;
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

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const f = acceptedFiles[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setActiveHeatmap(null);
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

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans">

      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">🩻</span>
            <div>
              <h1 className="text-xl font-bold text-blue-400 leading-tight">ChestVision AI</h1>
              <p className="text-gray-500 text-xs">Explainable Chest X-Ray Disease Detection</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden sm:block text-xs bg-blue-950 text-blue-300 border border-blue-800 px-3 py-1 rounded-full">
              EfficientNet-B0 · Multimodal
            </span>
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
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200
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
              <div className="rounded-2xl overflow-hidden bg-gray-900 border border-gray-800">
                <img src={preview} alt="X-Ray preview" className="w-full object-contain max-h-56 bg-black" />
                <div className="px-4 py-2 flex items-center justify-between">
                  <p className="text-gray-400 text-xs truncate max-w-[70%]">{file?.name}</p>
                  <p className="text-gray-600 text-xs">{file ? (file.size / 1024).toFixed(0) + ' KB' : ''}</p>
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
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 text-sm focus:outline-none focus:border-blue-500"
                    placeholder="Age"
                  />
                </div>
                <div>
                  <label className="text-gray-500 text-xs mb-1 block">Sex</label>
                  <select
                    value={patientSex}
                    onChange={e => setPatientSex(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 text-sm focus:outline-none focus:border-blue-500"
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
                bg-blue-600 hover:bg-blue-500 active:bg-blue-700
                disabled:bg-gray-800 disabled:text-gray-600 disabled:cursor-not-allowed text-white"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                  </svg>
                  Analyzing X-Ray...
                </span>
              ) : <span>🔍 Analyze X-Ray</span>}
            </button>

            {/* Error */}
            {error && (
              <div className="bg-red-950 border border-red-800 rounded-xl p-4">
                <p className="text-red-400 text-sm">❌ {error}</p>
              </div>
            )}

            {/* Scan info */}
            {result && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-1 text-xs">
                <p className="text-gray-500 font-semibold uppercase tracking-wider mb-2">Scan Info</p>
                <p className="text-gray-400"><span className="text-gray-600">ID: </span>{result.scan_id.slice(0, 8)}...</p>
                <p className="text-gray-400"><span className="text-gray-600">File: </span>{result.filename}</p>
                <p className="text-gray-400"><span className="text-gray-600">Patient: </span>{result.sex}, Age {result.age}</p>
                <p className="text-gray-400"><span className="text-gray-600">Time: </span>{result.report.generated_at}</p>
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
                <div className="text-6xl mb-4 opacity-30">📊</div>
                <p className="text-gray-600 text-lg font-medium">No analysis yet</p>
                <p className="text-gray-700 text-sm mt-2">Upload a chest X-ray and click Analyze</p>
              </div>
            ) : (
              <div className="space-y-4">

                {/* Tabs */}
                <div className="flex gap-1 bg-gray-900 p-1 rounded-xl border border-gray-800">
                  {tabs.map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex-1 py-2 px-2 rounded-lg text-xs font-medium transition-all duration-150
                        ${activeTab === tab.id ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-500 hover:text-gray-300'}`}
                    >
                      <span className="hidden sm:inline">{tab.emoji} </span>{tab.label}
                    </button>
                  ))}
                </div>

                {/* Tab: Predictions */}
                {activeTab === 'predictions' && (
                  <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 space-y-4">
                    <div className="flex items-center justify-between mb-2">
                      <h2 className="font-semibold text-gray-200">Disease Predictions</h2>
                      <span className="text-xs text-gray-600">threshold: 50%</span>
                    </div>
                    {result.report.all_predictions.map(p => {
                      const pct = Math.round(p.probability * 100);
                      const positive = p.probability >= 0.5;
                      return (
                        <div key={p.disease} className="space-y-1">
                          <div className="flex justify-between items-center">
                            <div className="flex items-center gap-2">
                              <span className={`text-xs px-2 py-0.5 rounded-full font-medium
                                ${positive ? 'bg-red-950 text-red-400 border border-red-900' : 'bg-gray-800 text-gray-500'}`}>
                                {positive ? 'POSITIVE' : 'negative'}
                              </span>
                              <span className={`text-sm font-medium ${getConfidenceColor(p.probability)}`}>{p.disease}</span>
                            </div>
                            <span className={`text-sm font-bold tabular-nums ${getConfidenceColor(p.probability)}`}>{pct}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-2">
                            <div className={`h-2 rounded-full transition-all duration-500 ${getBarColor(p.probability)}`} style={{ width: `${pct}%` }} />
                          </div>
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
                    <p className="text-gray-600 text-xs">Red/yellow regions indicate areas the model focused on for each prediction.</p>
                    {Object.keys(result.heatmaps).length > 1 && (
                      <div className="flex gap-2 flex-wrap">
                        {Object.keys(result.heatmaps).map(disease => (
                          <button key={disease} onClick={() => setActiveHeatmap(disease)}
                            className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-colors
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
                        className="text-xs bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors font-medium">
                        ⬇ Download PDF
                      </button>
                    </div>

                    <div className="grid grid-cols-2 gap-3 bg-gray-800 rounded-xl p-4 text-xs">
                      <div><p className="text-gray-600">Report ID</p><p className="text-gray-300 font-mono">{result.report.report_id}</p></div>
                      <div><p className="text-gray-600">Generated</p><p className="text-gray-300">{result.report.generated_at}</p></div>
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