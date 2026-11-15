import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { Result } from '../api';
import Results from './Results';

const prediction = (disease: string, probability: number, scores: [number, number, number | null]) => ({
  disease, probability, positive: probability >= 0.5,
  model_scores: { efficientnet_b0: scores[0], mobilenet_v2: scores[1], torchxrayvision: scores[2] },
  disagreement: Math.max(...(scores.filter(s => s !== null) as number[])) - Math.min(...(scores.filter(s => s !== null) as number[])),
});

const result: Result = {
  scan_id: 'scan-1', filename: 'xray.png', age: 60, sex: 'Unknown', original: 'AAAA', scan_db_id: null,
  // Deliberately not in display order: the page must put flagged findings first.
  predictions: [
    prediction('Edema', 0.2, [0.21, 0.19, 0.2]),
    prediction('Consolidation', 0.3, [0.05, 0.1, null]),
    prediction('Cardiomegaly', 0.71, [0.9, 0.7, 0.5]),
  ],
  heatmaps: { Cardiomegaly: 'BBBB' },
  report: {
    report_id: 'r1', generated_at: '', patient_id: '', patient_age: 60, patient_sex: 'Unknown', image_filename: 'xray.png',
    findings: 'Findings text', differential: '', impression: 'Impression text', recommendations: ['Step one'],
    disease_details: [{ disease: 'Cardiomegaly', probability: 0.71, description: 'About the heart', symptoms: ['s1'], causes: ['c1'], specialist: 'Cardiologist', region: 'Heart' }],
    disclaimer: 'Disclaimer text', llm_generated: false, all_predictions: [],
  },
};

const rows = () => screen.getAllByRole('listitem').filter(li => within(li).queryByRole('button', { expanded: false }) || within(li).queryByRole('button', { expanded: true }));

test('lists flagged findings first and summarises them without calling it a diagnosis', () => {
  render(<Results result={result} preview={null} onReset={() => {}} />);
  expect(rows().map(r => within(r).getAllByText(/Cardiomegaly|Consolidation|Edema/)[0].textContent))
    .toEqual(['Cardiomegaly', 'Consolidation', 'Edema']);
  expect(screen.getByText(/1 of 3 screened conditions scored above the review threshold: Cardiomegaly/)).toBeInTheDocument();
  expect(screen.getByText('71.0%')).toBeInTheDocument();
  expect(screen.queryByText(/^Detected$|^Normal$/)).not.toBeInTheDocument();
});

test('selecting a finding shows its heatmap, or says there is none', () => {
  render(<Results result={result} preview={null} onReset={() => {}} />);
  expect(screen.getByAltText('Heatmap for Cardiomegaly')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: /Edema/ }));
  expect(screen.queryByAltText(/Heatmap for/)).not.toBeInTheDocument();
  expect(screen.getByText('No heatmap was generated for Edema.')).toBeInTheDocument();
});

test('expanding a finding shows each model score, including a model that does not cover it', () => {
  render(<Results result={result} preview={null} onReset={() => {}} />);
  fireEvent.click(screen.getByRole('button', { name: /Consolidation/ }));
  expect(screen.getByText('Why this score')).toBeInTheDocument();
  expect(screen.getByText('Does not cover this condition')).toBeInTheDocument();
});

test('condition information opens in a panel and closes with Escape', () => {
  render(<Results result={result} preview={null} onReset={() => {}} />);
  fireEvent.click(screen.getByRole('button', { name: /Cardiomegaly/ }));
  fireEvent.click(screen.getByRole('button', { name: 'About this condition' }));
  expect(screen.getByRole('dialog')).toHaveTextContent('About the heart');
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

test('hides feedback when the backend did not store the scan', () => {
  render(<Results result={result} preview={null} onReset={() => {}} />);
  expect(screen.queryByText('Was this useful?')).not.toBeInTheDocument();
});
