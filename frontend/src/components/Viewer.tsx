import React, { useState } from 'react';

interface Props {
  original: string;            // image URL or data URL
  heatmap: string | null;      // data URL of the heatmap for the selected finding
  finding: string;
  hasAnyHeatmap: boolean;
  fromCache: boolean;
}

/** Dark image panel. The heatmap follows the finding selected in the list. */
export default function Viewer({ original, heatmap, finding, hasAnyHeatmap, fromCache }: Props) {
  const [show, setShow] = useState(true);
  const [opacity, setOpacity] = useState(100);
  const visible = show && !!heatmap;

  let caption: string;
  if (heatmap) {
    caption = `Heatmap for ${finding}. Warmer areas influenced the model's score more. It is not an outline of disease.`;
  } else if (!hasAnyHeatmap && fromCache) {
    caption = 'This image was analysed before, so the saved scores are shown. Heatmaps are not saved.';
  } else {
    caption = `No heatmap was generated for ${finding}.`;
  }

  return (
    <section aria-label="X-ray image" className="viewer bg-viewer-bg text-viewer-text rounded-card p-3">
      <div className="relative aspect-[4/5] lg:aspect-square rounded-control overflow-hidden bg-black">
        <img src={original} alt="Uploaded chest X-ray" className="absolute inset-0 w-full h-full object-contain" />
        {visible && (
          <img src={heatmap!} alt={`Heatmap for ${finding}`} style={{ opacity: opacity / 100 }}
               className="absolute inset-0 w-full h-full object-contain" />
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 px-1">
        <button
          type="button" aria-pressed={visible} disabled={!heatmap} onClick={() => setShow(s => !s)}
          className={`min-h-[44px] px-3.5 rounded-control text-[13px] font-semibold border transition-colors
            disabled:opacity-50 disabled:cursor-not-allowed
            ${visible ? 'bg-viewer-text text-viewer-bg border-viewer-text' : 'bg-viewer-panel text-viewer-text border-viewer-muted'}`}
        >
          Heatmap {visible ? 'on' : 'off'}
        </button>
        <label className="flex items-center gap-2 text-[13px] text-viewer-muted flex-1 min-w-[160px]">
          Strength
          <input type="range" min={10} max={100} step={5} value={opacity} disabled={!visible}
                 onChange={e => setOpacity(Number(e.target.value))}
                 className="flex-1 accent-[#7FD1C7] disabled:opacity-40" />
        </label>
      </div>
      <p className="mt-2 px-1 pb-1 text-[13px] leading-relaxed text-viewer-muted" aria-live="polite">{caption}</p>
    </section>
  );
}
