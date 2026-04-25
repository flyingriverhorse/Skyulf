import React from 'react';
import { CheckCircle, XCircle, Loader2, Clock, Ban } from 'lucide-react';

export type JobStatus =
  | 'completed'
  | 'COMPLETED'
  | 'succeeded'
  | 'failed'
  | 'FAILED'
  | 'running'
  | 'RUNNING'
  | 'processing'
  | 'pending'
  | 'PENDING'
  | 'cancelled'
  | 'CANCELLED'
  | 'idle';

export interface StatusBadgeProps {
  status: JobStatus | string;
  /** Hide the icon and only render the textual label. */
  iconOnly?: boolean;
  /** Hide the textual label and only render the icon. */
  textOnly?: boolean;
  /** Tailwind sizing override; defaults to text-xs. */
  className?: string;
}

interface StatusVisuals {
  label: string;
  classes: string;
  Icon: typeof CheckCircle;
  spin?: boolean;
}

const RESOLVE: Record<string, StatusVisuals> = {
  completed:  { label: 'Completed',  classes: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 border-green-200 dark:border-green-800',          Icon: CheckCircle },
  succeeded:  { label: 'Succeeded',  classes: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 border-green-200 dark:border-green-800',          Icon: CheckCircle },
  failed:     { label: 'Failed',     classes: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400 border-red-200 dark:border-red-800',                      Icon: XCircle },
  running:    { label: 'Running',    classes: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-800',                Icon: Loader2, spin: true },
  processing: { label: 'Processing', classes: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-800',                Icon: Loader2, spin: true },
  pending:    { label: 'Pending',    classes: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400 border-amber-200 dark:border-amber-800',          Icon: Clock },
  cancelled:  { label: 'Cancelled',  classes: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 border-slate-200 dark:border-slate-700',             Icon: Ban },
  idle:       { label: 'Idle',       classes: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 border-slate-200 dark:border-slate-700',             Icon: Clock },
};

const FALLBACK: StatusVisuals = {
  label: 'Unknown',
  classes: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 border-slate-200 dark:border-slate-700',
  Icon: Clock,
};

/** Pill-style badge for job/dataset statuses with consistent colors and iconography. */
export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, iconOnly, textOnly, className = '' }) => {
  const v = RESOLVE[status?.toLowerCase?.() ?? ''] ?? FALLBACK;
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-medium ${v.classes} ${className}`}
      title={v.label}
    >
      {!textOnly && <v.Icon size={12} className={v.spin ? 'animate-spin' : ''} />}
      {!iconOnly && <span className="capitalize">{v.label}</span>}
    </span>
  );
};
