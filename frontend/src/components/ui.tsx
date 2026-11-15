import React from 'react';

/* ---- Icons (inline SVG, inherit text colour) ------------------------------ */
type IconProps = { className?: string };
const svg = (path: React.ReactNode, filled = false) => ({ className = 'w-4 h-4' }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill={filled ? 'currentColor' : 'none'} stroke={filled ? 'none' : 'currentColor'}
       strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {path}
  </svg>
);

export const UploadIcon = svg(<><path d="M12 16V4" /><path d="M7 9l5-5 5 5" /><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" /></>);
export const DownloadIcon = svg(<><path d="M12 4v12" /><path d="M7 11l5 5 5-5" /><path d="M4 20h16" /></>);
export const ChevronIcon = svg(<path d="M6 9l6 6 6-6" />);
export const CloseIcon = svg(<><path d="M6 6l12 12" /><path d="M18 6L6 18" /></>);
export const AlertIcon = svg(<><path d="M12 4l9 16H3z" /><path d="M12 10v4" /><path d="M12 17.5v.01" /></>);
export const CheckIcon = svg(<path d="M5 12.5l4.5 4.5L19 7.5" />);
export const FileIcon = svg(<><path d="M7 3h7l5 5v13H7z" /><path d="M14 3v5h5" /></>);
export const TriangleMark = svg(<path d="M12 5l8 14H4z" />, true);
export const DiamondMark = svg(<path d="M12 3l9 9-9 9-9-9z" />, true);
export const DotMark = svg(<circle cx="12" cy="12" r="6" />, true);

export const LogoMark = ({ className = 'w-7 h-7' }: IconProps) => (
  <svg className={className} viewBox="0 0 32 32" aria-hidden="true">
    <rect width="32" height="32" rx="7" fill="#0F6B63" />
    <path d="M16 7v18M16 11c-4 0-7 3-7 8v4c3 0 5-1 7-3M16 11c4 0 7 3 7 8v4c-3 0-5-1-7-3"
          fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export const Spinner = ({ className = 'w-4 h-4' }: IconProps) => (
  <svg className={`spinner ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
    <path d="M21 12a9 9 0 00-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
  </svg>
);

/* ---- Button --------------------------------------------------------------- */
type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'md' | 'sm';
};

export function Button({ variant = 'secondary', size = 'md', className = '', type = 'button', ...rest }: ButtonProps) {
  const base = 'inline-flex items-center justify-center gap-2 rounded-control font-semibold transition-colors ' +
    'disabled:cursor-not-allowed ' + (size === 'md' ? 'min-h-[44px] px-4 text-sm' : 'min-h-[36px] px-3 text-[13px]');
  const look = {
    primary: 'bg-accent text-white hover:bg-accent-hover disabled:bg-neutral-bg disabled:text-ink-muted',
    secondary: 'bg-card text-ink border border-control hover:bg-accent-tint disabled:opacity-60',
    ghost: 'text-accent hover:bg-accent-tint disabled:opacity-60',
  }[variant];
  return <button type={type} className={`${base} ${look} ${className}`} {...rest} />;
}

/* ---- Status chip: always icon + word + colour ----------------------------- */
export type Status = 'flagged' | 'below' | 'disagree';

const STATUS: Record<Status, { label: string; cls: string; Icon: (p: IconProps) => React.ReactElement }> = {
  flagged: { label: 'Flagged for review', cls: 'bg-flag-bg text-flag-text', Icon: TriangleMark },
  below: { label: 'Below threshold', cls: 'bg-neutral-bg text-neutral-text', Icon: DotMark },
  disagree: { label: 'Models disagree', cls: 'bg-caution-bg text-caution-text', Icon: DiamondMark },
};

export function StatusChip({ status }: { status: Status }) {
  const { label, cls, Icon } = STATUS[status];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold whitespace-nowrap ${cls}`}>
      <Icon className="w-2.5 h-2.5" />
      {label}
    </span>
  );
}

/* ---- Inline message ------------------------------------------------------- */
export function Notice({ tone, title, children }: { tone: 'error' | 'info'; title?: string; children: React.ReactNode }) {
  const cls = tone === 'error' ? 'bg-flag-bg text-flag-text' : 'bg-accent-tint text-accent-hover';
  return (
    <div role={tone === 'error' ? 'alert' : 'status'} className={`flex gap-2.5 rounded-control px-3.5 py-3 text-[13px] leading-relaxed ${cls}`}>
      {tone === 'error' ? <AlertIcon className="w-4 h-4 mt-0.5 shrink-0" /> : <CheckIcon className="w-4 h-4 mt-0.5 shrink-0" />}
      <div>
        {title && <p className="font-semibold">{title}</p>}
        <div>{children}</div>
      </div>
    </div>
  );
}

export const formatScore = (p: number) => `${(p * 100).toFixed(1)}%`;
