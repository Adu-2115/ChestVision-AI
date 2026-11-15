import React, { useCallback, useMemo, useState } from 'react';
import { Result, errorMessage, fetchReportPdf } from '../api';
import ConditionDrawer from './ConditionDrawer';
import Feedback from './Feedback';
import FindingRow from './FindingRow';
import Viewer from './Viewer';
import { Button, ChevronIcon, DownloadIcon, Notice, Spinner } from './ui';

const png = (b64: string) => `data:image/png;base64,${b64}`;

function ReportSection({ title, children, open = false }: { title: string; children: React.ReactNode; open?: boolean }) {
  return (
    <details open={open} className="group border-t border-line">
      <summary className="flex items-center justify-between cursor-pointer list-none min-h-[44px] font-semibold">
        {title}
        <ChevronIcon className="w-4 h-4 text-ink-muted transition-transform group-open:rotate-180" />
      </summary>
      <div className="pb-4 text-ink-2 leading-relaxed max-w-[70ch] whitespace-pre-line">{children}</div>
    </details>
  );
}

export default function Results({ result, preview, onReset }: { result: Result; preview: string | null; onReset: () => void }) {
  // Flagged findings first, then by score.
  const findings = useMemo(
    () => [...result.predictions].sort((a, b) => Number(b.positive) - Number(a.positive) || b.probability - a.probability),
    [result.predictions]
  );
  const flagged = findings.filter(f => f.positive);
  const firstWithHeatmap = findings.find(f => result.heatmaps[f.disease]);

  const [selected, setSelected] = useState<string>((firstWithHeatmap || findings[0])?.disease ?? '');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [about, setAbout] = useState<string | null>(null);
  const [pdfBusy, setPdfBusy] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const closeDrawer = useCallback(() => setAbout(null), []);

  const details = result.report.disease_details || [];
  const aboutDetail = details.find(d => d.disease === about);
  const heatmap = result.heatmaps[selected];
  const original = result.original ? png(result.original) : (preview || '');

  const summary = flagged.length === 0
    ? `No condition scored above the review threshold among the ${findings.length} conditions this tool screens. ` +
      'This does not rule out disease, and the tool does not assess other conditions.'
    : `${flagged.length} of ${findings.length} screened conditions scored above the review threshold: ` +
      `${flagged.map(f => f.disease).join(', ')}. These are model scores to review with a clinician, not a diagnosis.`;

  const downloadPdf = async () => {
    setPdfBusy(true);
    setPdfError(null);
    try {
      const url = window.URL.createObjectURL(await fetchReportPdf(result));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `ChestVision_${result.scan_id}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => window.URL.revokeObjectURL(url), 10000);  // after the download has started
    } catch (e) {
      setPdfError(errorMessage(e, "The PDF couldn't be created. Try again."));
    } finally {
      setPdfBusy(false);
    }
  };

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
        <div className="min-w-0">
          <h2 className="font-serif text-[28px] leading-tight font-semibold">Your result</h2>
          <p className="mt-1 text-[13px] text-ink-muted break-words">
            {result.filename} · Age {result.age}, sex {result.sex.toLowerCase()}
            {result.from_cache && <span className="ml-2 rounded-full bg-accent-tint text-accent-hover px-2 py-0.5 font-semibold">Saved result</span>}
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={downloadPdf} disabled={pdfBusy}>
            {pdfBusy ? <Spinner /> : <DownloadIcon />} Download PDF
          </Button>
          <Button variant="primary" onClick={onReset}>New scan</Button>
        </div>
      </div>
      {pdfError && <div className="mt-3"><Notice tone="error">{pdfError}</Notice></div>}

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-5 lg:sticky lg:top-20">
          <Viewer original={original} heatmap={heatmap ? png(heatmap) : null} finding={selected}
                  hasAnyHeatmap={Object.keys(result.heatmaps).length > 0} fromCache={!!result.from_cache} />
        </div>

        <div className="lg:col-span-7 space-y-6">
          <section className="card p-4 md:p-6">
            <h3 className="text-xl font-semibold">Summary</h3>
            <p className="mt-2 text-base leading-relaxed max-w-[70ch]">{summary}</p>
          </section>

          <section className="card p-4 md:p-6">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="text-xl font-semibold">Findings</h3>
              <p className="text-[13px] text-ink-muted">{findings.length} conditions screened · review threshold 50%</p>
            </div>
            <p className="mt-1 text-[13px] text-ink-muted">Select a finding to see its heatmap and each model's score.</p>
            <ul className="mt-4 space-y-2.5">
              {findings.map(f => (
                <FindingRow
                  key={f.disease}
                  prediction={f}
                  selected={selected === f.disease}
                  expanded={expanded === f.disease}
                  hasDetail={details.some(d => d.disease === f.disease)}
                  onSelect={() => { setSelected(f.disease); setExpanded(expanded === f.disease ? null : f.disease); }}
                  onAbout={() => setAbout(f.disease)}
                />
              ))}
            </ul>
          </section>

          <section className="card p-4 md:p-6">
            <h3 className="text-xl font-semibold">Written summary</h3>
            <p className="mt-1 mb-4 text-[13px] text-ink-muted">
              {result.report.llm_generated
                ? 'Drafted by a language model from the scores above. It can contain mistakes.'
                : 'Filled in from a standard template using the scores above.'}
            </p>
            <ReportSection title="Impression" open>{result.report.impression}</ReportSection>
            <ReportSection title="Findings">{result.report.findings}</ReportSection>
            {result.report.differential && <ReportSection title="Other possibilities">{result.report.differential}</ReportSection>}
            {result.report.recommendations?.length > 0 && (
              <ReportSection title="Suggested next steps">
                <ul className="list-disc pl-5 space-y-1 whitespace-normal">
                  {result.report.recommendations.map((r, i) => <li key={i}>{r}</li>)}
                </ul>
              </ReportSection>
            )}
            {result.report.disclaimer && (
              <p className="border-t border-line pt-4 text-[13px] text-ink-muted leading-relaxed max-w-[70ch]">{result.report.disclaimer}</p>
            )}
          </section>

          {result.scan_db_id ? <Feedback scanDbId={result.scan_db_id} /> : null}
        </div>
      </div>

      {aboutDetail && <ConditionDrawer detail={aboutDetail} onClose={closeDrawer} />}
    </div>
  );
}
