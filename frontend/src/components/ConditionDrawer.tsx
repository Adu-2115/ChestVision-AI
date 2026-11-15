import React, { useEffect, useRef } from 'react';
import { DiseaseDetail } from '../api';
import { CloseIcon } from './ui';

/** Side panel with background information on one condition. Esc or the backdrop closes it. */
export default function ConditionDrawer({ detail, onClose }: { detail: DiseaseDetail; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      // The close button is the only control in the panel, so Tab stays on it.
      if (e.key === 'Tab') { e.preventDefault(); closeRef.current?.focus(); }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      previous?.focus();
    };
  }, [onClose]);

  const list = (title: string, items: string[]) => items.length > 0 && (
    <section>
      <h3 className="text-[13px] font-semibold text-ink-2">{title}</h3>
      <ul className="mt-1.5 space-y-1 list-disc pl-5 leading-relaxed">
        {items.map(i => <li key={i}>{i}</li>)}
      </ul>
    </section>
  );

  return (
    <div className="fixed inset-0 z-30 flex justify-end">
      <div className="absolute inset-0 bg-ink/40" onClick={onClose} aria-hidden="true" />
      <aside role="dialog" aria-modal="true" aria-labelledby="drawer-title"
             className="relative w-full max-w-md h-full bg-card border-l border-line overflow-y-auto">
        <div className="sticky top-0 bg-card border-b border-line px-5 h-14 flex items-center justify-between gap-3">
          <h2 id="drawer-title" className="text-base font-semibold truncate">{detail.disease}</h2>
          <button ref={closeRef} type="button" onClick={onClose} aria-label="Close"
                  className="w-11 h-11 -mr-2 inline-flex items-center justify-center rounded-control hover:bg-accent-tint">
            <CloseIcon className="w-5 h-5" />
          </button>
        </div>
        <div className="px-5 py-5 space-y-5">
          <p className="leading-relaxed">{detail.description}</p>
          {detail.region && <p className="text-ink-2"><span className="font-semibold">Usually seen in:</span> {detail.region}</p>}
          {list('Common symptoms', detail.symptoms)}
          {list('Common causes', detail.causes)}
          {detail.specialist && <p className="text-ink-2"><span className="font-semibold">Specialist:</span> {detail.specialist}</p>}
          <p className="text-[13px] text-ink-muted leading-relaxed border-t border-line pt-4">
            General background, not advice about this image. Discuss symptoms and next steps with a clinician.
          </p>
        </div>
      </aside>
    </div>
  );
}
