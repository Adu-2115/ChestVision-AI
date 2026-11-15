import React from 'react';
import { LogoMark } from './ui';

export default function Header() {
  return (
    <header className="sticky top-0 z-20 bg-card border-b border-line">
      <div className="max-w-[1280px] mx-auto h-14 px-4 md:px-6 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <LogoMark className="w-7 h-7 shrink-0" />
          <h1 className="text-base font-semibold tracking-tight truncate">ChestVision AI</h1>
        </div>
        <p className="shrink-0 rounded-full bg-caution-bg text-caution-chip px-3 py-1 text-xs font-semibold">
          Research preview<span className="hidden sm:inline">, not a diagnosis</span>
        </p>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="max-w-[1280px] mx-auto px-4 md:px-6 py-8 text-[13px] text-ink-muted">
      Not a medical device. Scores come from three models (EfficientNet-B0, MobileNetV2, TorchXRayVision) reading
      one image, and can be wrong. Review any result with a qualified clinician.
    </footer>
  );
}
