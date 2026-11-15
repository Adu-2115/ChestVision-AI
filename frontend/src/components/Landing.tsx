import React from 'react';
import { ChevronIcon, StatusChip } from './ui';

// The five conditions the deployed models score.
const CONDITIONS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural effusion'];

const STEPS = [
  { title: 'Upload', text: 'Choose a front-view chest X-ray as JPG, PNG or DICOM.' },
  { title: 'Analyse', text: 'Three models score the image for five conditions.' },
  { title: 'Review', text: 'See each score, where the models looked, and where they disagree.' },
];

const LIMITS = [
  'Reads one front-view image. It knows nothing else about the person.',
  'Scores five conditions only. Anything else is not assessed.',
  'Scores can be wrong, and the models sometimes disagree with each other.',
  'Not a medical device. A clinician must review the image and the result.',
];

const FAQ = [
  { q: 'Is this a diagnosis?',
    a: 'No. The result is a set of model scores for one image. Only a clinician who knows your situation can diagnose.' },
  { q: 'What happens to my image?',
    a: 'The image file is deleted from the server when the analysis finishes. The scores and written summary may be stored.' },
  { q: 'Which files work?',
    a: 'A front-view chest X-ray as JPG, PNG or DICOM, up to 15 MB. Photos of a screen or a printed film may give unreliable results.' },
  { q: 'How accurate is it?',
    a: 'Accuracy figures for the models running here have not been published on this site yet, so treat every score with caution.' },
];

const AnalyzeLink = ({ className = '' }: { className?: string }) => (
  <a href="#/analyze"
     className={`inline-flex items-center justify-center min-h-[44px] px-5 rounded-control bg-accent hover:bg-accent-hover text-white font-semibold transition-colors ${className}`}>
    Analyze an X-ray
  </a>
);

function ExampleBar({ value, flagged }: { value: number; flagged: boolean }) {
  return (
    <div className="relative h-2 rounded-full bg-neutral-bg" aria-hidden="true">
      <div className={`h-2 rounded-full ${flagged ? 'bg-flag-bar' : 'bg-neutral-bar'}`} style={{ width: `${value}%` }} />
      <div className="absolute -top-1 left-1/2 w-0.5 h-4 bg-ink" />
    </div>
  );
}

export default function Landing() {
  return (
    <div className="space-y-16 md:space-y-20">
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center pt-4 md:pt-8">
        <div className="lg:col-span-7">
          <h2 className="font-serif text-4xl md:text-[44px] leading-[1.1] font-semibold max-w-[16ch]">
            A second look at a chest X-ray
          </h2>
          <p className="mt-4 text-base md:text-lg text-ink-2 leading-relaxed max-w-[52ch]">
            Upload a chest X-ray and three models score it for five common findings. You see each score,
            where the models looked, and where they disagree.
          </p>
          <AnalyzeLink className="mt-6" />
          <ul className="mt-6 flex flex-wrap gap-x-6 gap-y-2 text-[13px] text-ink-muted">
            <li>Research preview, not a diagnosis</li>
            <li>Image file deleted after analysis</li>
            <li>Shows where models disagree</li>
          </ul>
        </div>

        <div className="lg:col-span-5">
          <div className="card p-4 md:p-5">
            <p className="text-xs font-semibold text-ink-muted">Example, not a real result</p>
            <ul className="mt-3 space-y-4">
              {[{ name: 'Cardiomegaly', score: '71.3%', value: 71, flagged: true },
                { name: 'Edema', score: '31.2%', value: 31, flagged: false },
                { name: 'Atelectasis', score: '9.1%', value: 9, flagged: false }].map(r => (
                <li key={r.name}>
                  <div className="flex items-center justify-between gap-3 mb-2">
                    <span className="flex flex-wrap items-center gap-2">
                      <span className="font-semibold">{r.name}</span>
                      <StatusChip status={r.flagged ? 'flagged' : 'below'} />
                    </span>
                    <span className="font-mono tabular">{r.score}</span>
                  </div>
                  <ExampleBar value={r.value} flagged={r.flagged} />
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section aria-labelledby="how">
        <h3 id="how" className="text-xl font-semibold">How it works</h3>
        <ol className="mt-5 grid grid-cols-1 md:grid-cols-3 gap-4">
          {STEPS.map((s, i) => (
            <li key={s.title} className="card p-5">
              <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-accent-tint text-accent-hover font-semibold font-mono">{i + 1}</span>
              <p className="mt-3 font-semibold text-base">{s.title}</p>
              <p className="mt-1 text-ink-2 leading-relaxed">{s.text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="scope" className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <h3 id="scope" className="text-xl font-semibold md:col-span-2">What it can and cannot do</h3>
        <div className="card p-5">
          <p className="font-semibold text-base">Conditions it scores</p>
          <ul className="mt-3 space-y-1.5 list-disc pl-5 text-ink-2">
            {CONDITIONS.map(c => <li key={c}>{c}</li>)}
          </ul>
        </div>
        <div className="card p-5">
          <p className="font-semibold text-base">Limits</p>
          <ul className="mt-3 space-y-1.5 list-disc pl-5 text-ink-2 leading-relaxed">
            {LIMITS.map(l => <li key={l}>{l}</li>)}
          </ul>
        </div>
      </section>

      <section aria-labelledby="faq" className="max-w-[760px]">
        <h3 id="faq" className="text-xl font-semibold">Questions</h3>
        <div className="mt-4 border-b border-line">
          {FAQ.map(f => (
            <details key={f.q} className="group border-t border-line">
              <summary className="flex items-center justify-between gap-3 cursor-pointer list-none min-h-[48px] font-semibold">
                {f.q}
                <ChevronIcon className="w-4 h-4 shrink-0 text-ink-muted transition-transform group-open:rotate-180" />
              </summary>
              <p className="pb-4 text-ink-2 leading-relaxed">{f.a}</p>
            </details>
          ))}
        </div>
        <AnalyzeLink className="mt-8" />
      </section>
    </div>
  );
}
