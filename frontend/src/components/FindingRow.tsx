import React from 'react';
import { MODEL_LABELS, ModelScores, Prediction } from '../api';
import { Button, ChevronIcon, StatusChip, formatScore } from './ui';

const AGREEMENT = [
  { below: 0.15, label: 'Models agree', help: 'All models gave similar scores.' },
  { below: 0.35, label: 'Partial agreement', help: 'Models differ somewhat. Interpret with extra caution.' },
  { below: Infinity, label: 'Models disagree', help: 'Models give very different scores. This image needs direct review by a clinician.' },
];
export const modelsDisagree = (p: Prediction) => (p.disagreement ?? 0) >= 0.35;

function ScoreBar({ value, flagged, label }: { value: number; flagged: boolean; label: string }) {
  return (
    <div role="img" aria-label={label} className="relative h-2 rounded-full bg-neutral-bg">
      <div className={`h-2 rounded-full ${flagged ? 'bg-flag-bar' : 'bg-neutral-bar'}`} style={{ width: `${Math.round(value * 100)}%` }} />
      <div className="absolute -top-1 left-1/2 w-0.5 h-4 bg-ink" />
    </div>
  );
}

interface Props {
  prediction: Prediction;
  selected: boolean;
  expanded: boolean;
  hasDetail: boolean;
  onSelect: () => void;
  onAbout: () => void;
}

export default function FindingRow({ prediction: p, selected, expanded, hasDetail, onSelect, onAbout }: Props) {
  const id = `why-${p.disease.replace(/\W+/g, '-')}`;
  const scores = p.model_scores;
  const present = scores ? (Object.values(scores).filter(v => v !== null && v !== undefined) as number[]) : [];
  const agreement = p.disagreement === undefined ? null : AGREEMENT.find(a => p.disagreement! < a.below)!;

  return (
    <li className={`rounded-control border ${selected ? 'border-accent bg-accent-tint/40' : 'border-line bg-card'}`}>
      <button type="button" onClick={onSelect} aria-expanded={expanded} aria-controls={id}
              className="w-full text-left px-3.5 py-3 rounded-control">
        <span className="flex items-start justify-between gap-3">
          <span className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 min-w-0">
            <span className="text-[15px] font-semibold">{p.disease}</span>
            <StatusChip status={p.positive ? 'flagged' : 'below'} />
            {modelsDisagree(p) && <StatusChip status="disagree" />}
          </span>
          <span className="flex items-center gap-2 shrink-0">
            <span className="font-mono tabular text-[15px] font-medium">{formatScore(p.probability)}</span>
            <ChevronIcon className={`w-4 h-4 text-ink-muted transition-transform ${expanded ? 'rotate-180' : ''}`} />
          </span>
        </span>
        <span className="block mt-3">
          <ScoreBar value={p.probability} flagged={p.positive}
                    label={`Model score ${formatScore(p.probability)}. Review threshold 50%.`} />
        </span>
      </button>

      {expanded && (
        <div id={id} className="px-3.5 pb-4 pt-1 border-t border-line space-y-3">
          <h4 className="pt-3 text-[13px] font-semibold text-ink-2">Why this score</h4>
          {scores ? (
            <ul className="space-y-2">
              {(Object.keys(MODEL_LABELS) as (keyof ModelScores)[]).map(key => {
                const s = scores[key];
                const covered = s !== null && s !== undefined;
                return (
                  <li key={key} className="grid grid-cols-[130px_1fr_56px] items-center gap-3 text-[13px]">
                    <span className="text-ink-2">{MODEL_LABELS[key]}</span>
                    {covered ? (
                      <span className="h-1.5 rounded-full bg-neutral-bg" aria-hidden="true">
                        <span className="block h-1.5 rounded-full bg-neutral-bar" style={{ width: `${Math.round(s! * 100)}%` }} />
                      </span>
                    ) : <span className="text-ink-muted">Does not cover this condition</span>}
                    <span className="font-mono tabular text-right">{covered ? formatScore(s!) : ''}</span>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="text-[13px] text-ink-muted">Per-model scores are not available for this result.</p>
          )}
          {agreement && (
            <p className="text-[13px] text-ink-2 leading-relaxed">
              <span className="font-semibold">{agreement.label}.</span> {agreement.help}
              {present.length > 1 && (
                <> Model range <span className="font-mono tabular">{formatScore(Math.min(...present))}</span> to{' '}
                  <span className="font-mono tabular">{formatScore(Math.max(...present))}</span>.</>
              )}
            </p>
          )}
          {hasDetail && <Button variant="ghost" size="sm" className="-ml-3" onClick={onAbout}>About this condition</Button>}
        </div>
      )}
    </li>
  );
}
