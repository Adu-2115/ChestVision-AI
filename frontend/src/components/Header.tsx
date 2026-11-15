import React from 'react';
import { LogoMark } from './ui';

export default function Header({ onAnalyze }: { onAnalyze: boolean }) {
  return (
    <header className="sticky top-0 z-20 bg-card border-b border-line">
      <div className="max-w-[1280px] mx-auto h-14 px-4 md:px-6 flex items-center justify-between gap-3">
        <a href="#/" className="flex items-center gap-2.5 min-w-0 rounded-control" aria-label="ChestVision AI home">
          <LogoMark className="w-7 h-7 shrink-0" />
          <h1 className="text-base font-semibold tracking-tight truncate">ChestVision AI</h1>
        </a>
        <div className="flex items-center gap-3 shrink-0">
          {!onAnalyze && (
            <a href="#/analyze" className="hidden sm:inline-flex items-center min-h-[36px] px-3 rounded-control text-[13px] font-semibold text-accent hover:bg-accent-tint">
              Analyze
            </a>
          )}
          <p className="rounded-full bg-caution-bg text-caution-chip px-3 py-1 text-xs font-semibold">
            Research preview<span className="hidden sm:inline">, not a diagnosis</span>
          </p>
        </div>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="border-t border-line mt-8">
      <p className="max-w-[1280px] mx-auto px-4 md:px-6 py-6 text-[13px] text-ink-muted leading-relaxed">
        Not a medical device. Scores come from three models (EfficientNet-B0, MobileNetV2, TorchXRayVision) reading
        one image, and can be wrong. Review any result with a qualified clinician.
      </p>
    </footer>
  );
}
