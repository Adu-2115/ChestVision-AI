import React from 'react';
import { useDropzone } from 'react-dropzone';
import { Button, FileIcon, Notice, Spinner, UploadIcon } from './ui';

const SEX_OPTIONS = [
  { value: 'Male', label: 'Male' },
  { value: 'Female', label: 'Female' },
  { value: 'Unknown', label: 'Prefer not to say' },
];

interface Props {
  file: File | null;
  preview: string | null;
  age: string;
  sex: string;
  loading: boolean;
  error: string | null;
  onFile: (file: File) => void;
  onRemove: () => void;
  onAge: (age: string) => void;
  onSex: (sex: string) => void;
  onAnalyze: () => void;
}

const fileSize = (bytes: number) =>
  bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;

export default function UploadScreen(p: Props) {
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop: files => files[0] && p.onFile(files[0]),
    accept: { 'image/*': ['.jpg', '.jpeg', '.png'], 'application/dicom': ['.dcm'] },
    multiple: false,
    noClick: true,
    noKeyboard: true,
    disabled: p.loading,
  });
  const canPreview = !!p.preview && !!p.file && p.file.type.startsWith('image/');
  const ageNumber = Number(p.age);
  const ageInvalid = p.age !== '' && (!Number.isFinite(ageNumber) || ageNumber < 0 || ageNumber > 120);

  return (
    <div className="max-w-[640px] mx-auto">
      <h2 className="font-serif text-[28px] leading-tight font-semibold">Upload a chest X-ray</h2>
      <p className="mt-2 text-base text-ink-2">
        Three models score the image for common chest findings and show where they looked.
      </p>

      <div className="card mt-6 p-4 md:p-6 space-y-5">
        {!p.file ? (
          <div
            {...getRootProps()}
            className={`rounded-control border-2 border-dashed px-4 py-10 text-center transition-colors
              ${isDragActive ? 'border-accent bg-accent-tint' : 'border-control bg-app'}`}
          >
            <input {...getInputProps()} aria-label="Choose a chest X-ray file" />
            <UploadIcon className="w-7 h-7 mx-auto text-accent" />
            <p className="mt-3 text-base font-semibold">
              {isDragActive ? 'Drop the image here' : 'Drop a chest X-ray here'}
            </p>
            <Button variant="secondary" className="mt-3" onClick={open}>Choose a file</Button>
            <p className="mt-3 text-[13px] text-ink-muted">JPG, PNG or DICOM, up to 15 MB</p>
          </div>
        ) : (
          <div>
            <div className="flex items-center gap-3">
              <div className="w-16 h-16 shrink-0 rounded-control bg-viewer-bg overflow-hidden flex items-center justify-center text-viewer-muted">
                {canPreview
                  ? <img src={p.preview!} alt="" className="w-full h-full object-cover" />
                  : <FileIcon className="w-6 h-6" />}
              </div>
              <div className="min-w-0 flex-1">
                <p className="font-semibold truncate">{p.file.name}</p>
                <p className="text-[13px] text-ink-muted tabular">{fileSize(p.file.size)}</p>
              </div>
              <Button variant="ghost" size="sm" onClick={p.onRemove} disabled={p.loading}>Remove</Button>
            </div>
            {p.loading && (
              <div className="mt-4" role="status" aria-live="polite">
                <div className="h-1.5 rounded-full bg-neutral-bg overflow-hidden">
                  <div className="progress-indeterminate h-full w-2/5 rounded-full bg-accent" />
                </div>
                <p className="mt-2 text-[13px] text-ink-muted">
                  Analysing the image. This can take a minute or more if the service has been idle.
                </p>
              </div>
            )}
          </div>
        )}

        <details className="group rounded-control border border-line">
          <summary className="flex items-center justify-between gap-3 cursor-pointer list-none px-3.5 min-h-[44px] font-semibold">
            Add age and sex (optional)
            <span className="text-ink-muted text-[13px] font-normal group-open:hidden">Age {p.age || 60}, {p.sex === 'Unknown' ? 'sex not given' : p.sex.toLowerCase()}</span>
          </summary>
          <div className="px-3.5 pb-4 space-y-4">
            <p className="text-[13px] text-ink-muted leading-relaxed">
              Two of the three models use age and sex as extra context. If you leave these as they are, the tool
              uses age 60 and sex unknown, and results may be less accurate.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-[120px_1fr] gap-4">
              <div>
                <label htmlFor="age" className="block text-[13px] font-semibold mb-1.5">Age in years</label>
                <input
                  id="age" type="number" inputMode="numeric" min={0} max={120}
                  value={p.age} onChange={e => p.onAge(e.target.value)} disabled={p.loading}
                  aria-invalid={ageInvalid} aria-describedby={ageInvalid ? 'age-error' : undefined}
                  className="w-full h-11 rounded-control border border-control bg-card px-3 font-mono tabular"
                />
                {ageInvalid && <p id="age-error" className="mt-1 text-[13px] text-flag-text">Enter an age between 0 and 120.</p>}
              </div>
              <fieldset>
                <legend className="block text-[13px] font-semibold mb-1.5">Sex</legend>
                <div className="flex flex-wrap gap-2">
                  {SEX_OPTIONS.map(o => (
                    <label key={o.value}
                      className={`inline-flex items-center min-h-[44px] px-3.5 rounded-control border cursor-pointer text-sm
                        has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent
                        ${p.sex === o.value ? 'border-accent bg-accent-tint text-accent-hover font-semibold' : 'border-control bg-card'}`}>
                      <input type="radio" name="sex" value={o.value} checked={p.sex === o.value}
                             onChange={() => p.onSex(o.value)} disabled={p.loading} className="sr-only" />
                      {o.label}
                    </label>
                  ))}
                </div>
              </fieldset>
            </div>
          </div>
        </details>

        <ul className="space-y-1.5 text-[13px] text-ink-muted leading-relaxed list-disc pl-5">
          <li>Use a front-view chest X-ray. Photos of screens or printed films may give unreliable results.</li>
          <li>Remove names and ID numbers from the image before uploading.</li>
        </ul>

        {p.error && <Notice tone="error" title="The analysis didn't run">{p.error}</Notice>}

        <Button variant="primary" className="w-full" onClick={p.onAnalyze} disabled={!p.file || p.loading || ageInvalid}>
          {p.loading ? <><Spinner /> Analysing…</> : 'Analyze X-ray'}
        </Button>
      </div>
    </div>
  );
}
