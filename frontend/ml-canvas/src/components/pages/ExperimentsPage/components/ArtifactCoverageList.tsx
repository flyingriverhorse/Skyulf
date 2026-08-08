import React from 'react';
import { CheckCircle2, Clock, XCircle, MinusCircle, type LucideIcon } from 'lucide-react';
import type { ArtifactStatus } from '../utils/artifactCoverage';

export interface ArtifactCoverageEntry {
  jobId: string;
  /** Display label for the run, e.g. "random_forest (a1b2c3d4)". */
  label: string;
  status: ArtifactStatus;
  reason: string;
}

interface Props {
  entries: ArtifactCoverageEntry[];
}

// Icon + text pairing per status so availability never depends on color
// alone (a colorblind user or a printed/grayscale screenshot still reads
// "Available"/"Not computed"/"Unsupported"/"Failed" from the icon+label).
const STATUS_META: Record<ArtifactStatus, { label: string; icon: LucideIcon; className: string }> = {
  available: { label: 'Available', icon: CheckCircle2, className: 'text-emerald-600 dark:text-emerald-400' },
  not_computed: { label: 'Not computed', icon: Clock, className: 'text-amber-600 dark:text-amber-400' },
  unsupported: { label: 'Unsupported', icon: MinusCircle, className: 'text-gray-400 dark:text-gray-500' },
  failed: { label: 'Failed', icon: XCircle, className: 'text-red-600 dark:text-red-400' },
};

/**
 * Always-visible per-run availability list for an explainability/
 * segmentation artifact. Names, for every selected run, whether the
 * artifact is available, not yet computed, unsupported for that run's
 * task, or failed — and why — instead of a chart/tab silently disappearing
 * when data is partial (UX finding EXP-003).
 */
export const ArtifactCoverageList: React.FC<Props> = ({ entries }) => {
  if (entries.length === 0) return null;

  return (
    <div
      role="table"
      aria-label="Artifact availability by run"
      className="rounded-lg border border-gray-200 dark:border-gray-700 divide-y divide-gray-100 dark:divide-gray-800 bg-white dark:bg-gray-800 text-xs"
    >
      {entries.map((entry) => {
        const meta = STATUS_META[entry.status];
        const Icon = meta.icon;
        return (
          <div key={entry.jobId} role="row" className="flex items-start gap-2 px-3 py-2">
            <span className={`inline-flex items-center gap-1 font-medium shrink-0 ${meta.className}`}>
              <Icon className="w-3.5 h-3.5" aria-hidden="true" />
              {meta.label}
            </span>
            <span className="font-mono text-gray-700 dark:text-gray-300 shrink-0">{entry.label}</span>
            <span className="text-gray-500 dark:text-gray-400">{entry.reason}</span>
          </div>
        );
      })}
    </div>
  );
};
