import React, { useState } from 'react';
import { errorMessage, sendFeedback } from '../api';
import { Button, Notice } from './ui';

/** One line under the results; "Something looks wrong" opens a short form in place. */
export default function Feedback({ scanDbId }: { scanDbId: number }) {
  const [stage, setStage] = useState<'ask' | 'form' | 'done'>('ask');
  const [comments, setComments] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (useful: boolean) => {
    setBusy(true);
    setError(null);
    try {
      await sendFeedback(scanDbId, useful, comments);
      setStage('done');
    } catch (e) {
      setError(errorMessage(e, "Your feedback wasn't sent. Try again."));
    } finally {
      setBusy(false);
    }
  };

  if (stage === 'done') {
    return <Notice tone="info">Thanks. Your feedback helps improve the tool. It does not change this result.</Notice>;
  }
  return (
    <section className="card p-4 md:p-5 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-semibold">Was this useful?</h3>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => submit(true)} disabled={busy}>Yes</Button>
          <Button size="sm" onClick={() => setStage('form')} disabled={busy} aria-expanded={stage === 'form'}>
            Something looks wrong
          </Button>
        </div>
      </div>
      {stage === 'form' && (
        <div className="space-y-2">
          <label htmlFor="feedback" className="block text-[13px] font-semibold">What looks wrong? (optional)</label>
          <textarea id="feedback" rows={3} value={comments} onChange={e => setComments(e.target.value)}
                    className="w-full rounded-control border border-control bg-card px-3 py-2 leading-relaxed" />
          <Button variant="primary" size="sm" onClick={() => submit(false)} disabled={busy}>
            {busy ? 'Sending…' : 'Send feedback'}
          </Button>
        </div>
      )}
      {error && <Notice tone="error">{error}</Notice>}
    </section>
  );
}
