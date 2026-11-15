import axios from 'axios';

export const API_URL = process.env.REACT_APP_API_URL || 'https://adu2115-chessvision-api.hf.space';

export interface ModelScores {
  efficientnet_b0: number;
  mobilenet_v2: number;
  torchxrayvision: number | null;
}

export interface Prediction {
  disease: string;
  probability: number;
  positive: boolean;
  model_scores?: ModelScores;
  disagreement?: number;
  n_models_used?: number;
}

export interface DiseaseDetail {
  disease: string;
  probability: number;
  description: string;
  symptoms: string[];
  causes: string[];
  specialist: string;
  region: string;
}

export interface Report {
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

export interface Result {
  scan_id: string;
  filename: string;
  age: number;
  sex: string;
  predictions: Prediction[];
  heatmaps: Record<string, string>;
  original: string;
  report: Report;
  scan_db_id?: number | null;
  report_token?: string | null;
  from_cache?: boolean;
}

export const MODEL_LABELS: Record<keyof ModelScores, string> = {
  efficientnet_b0: 'EfficientNet-B0',
  mobilenet_v2: 'MobileNetV2',
  torchxrayvision: 'TorchXRayVision',
};

/** A sentence the user can act on, for any failed request. */
export function errorMessage(e: any, fallback: string): string {
  const detail = e?.response?.data?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (e?.code === 'ECONNABORTED') return 'The analysis took too long. Check your connection and try again.';
  if (!e?.response) return "Couldn't reach the analysis service. Check your connection and try again.";
  return fallback;
}

export async function analyze(file: File, age: number, sex: string): Promise<Result> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('age', age.toString());
  formData.append('sex', sex);
  const res = await axios.post(`${API_URL}/api/predict`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  });
  return res.data;
}

export async function sendFeedback(scanDbId: number, isCorrect: boolean, comments?: string): Promise<void> {
  await axios.post(`${API_URL}/api/feedback`, isCorrect
    ? { scan_db_id: scanDbId, is_correct: true }
    : { scan_db_id: scanDbId, is_correct: false, corrected_diagnosis: null, comments: comments || null });
}

/**
 * The newer backend builds the PDF from the stored scan and needs a token;
 * the older one builds it from the report object. Use whichever this result supports.
 */
export async function fetchReportPdf(result: Result): Promise<Blob> {
  const res = result.scan_db_id && result.report_token
    ? await axios.get(`${API_URL}/api/scans/${result.scan_db_id}/report.pdf`,
        { params: { token: result.report_token }, responseType: 'blob' })
    : await axios.post(`${API_URL}/api/report/generate`, result.report, { responseType: 'blob' });
  return new Blob([res.data]);
}
