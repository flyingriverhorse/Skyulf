import { AlertTriangle, XCircle } from 'lucide-react';
import type { GraphValidationIssue } from '../../../../core/store/useGraphStore';

interface ValidationBannerProps {
  validationIssues: GraphValidationIssue[];
  validationHeadingId: string;
  openIssue: (issue: GraphValidationIssue) => void;
}

/** Keep validation details outside the count-only live region. */
export function ValidationBanner({ validationIssues, validationHeadingId, openIssue }: ValidationBannerProps) {
  return validationIssues.length > 0 && (
    <section
      className="m-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900/40 dark:bg-red-950/20 dark:text-red-100"
      aria-labelledby={validationHeadingId}
    >
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-600 dark:text-red-400" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <p id={validationHeadingId} className="font-semibold">Validation issues</p>
          <p className="mt-0.5 text-xs text-red-700 dark:text-red-200">
            Fix one of the items below, then run preview again.
          </p>
          {/* Only the count is announced: the list recomputes on every graph
              edit, so a live region around it would re-read every issue on
              each keystroke. */}
          <p className="sr-only" role="status" aria-atomic="true">
            {validationIssues.length === 1
              ? '1 validation issue blocking preview'
              : `${validationIssues.length} validation issues blocking preview`}
          </p>
          <ul className="mt-3 space-y-2">
            {validationIssues.map((issue) => (
              <li key={`${issue.nodeId}-${issue.category}-${issue.message}`}>
                <button
                  type="button"
                  onClick={() => openIssue(issue)}
                  className="w-full rounded-md border border-red-200 bg-white/80 px-3 py-2 text-left transition-colors hover:bg-red-100 dark:border-red-900/40 dark:bg-slate-950/30 dark:hover:bg-red-950/40"
                >
                  <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wide text-red-700 dark:text-red-300">
                    <span className="rounded bg-red-100 px-1.5 py-0.5 dark:bg-red-950/50">{issue.category}</span>
                    <span className="font-semibold normal-case tracking-normal">{issue.nodeLabel}</span>
                  </div>
                  <p className="mt-1 text-sm text-slate-800 dark:text-slate-100">{issue.message}</p>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );


}

/** Announce the last failed preview independently of graph validation. */
export function RunErrorBanner({ lastRunError }: { lastRunError: string | null }) {
  return lastRunError && (
    <section
      className="m-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900/40 dark:bg-red-950/20 dark:text-red-100"
      role="alert"
      aria-atomic="true"
    >
      <div className="flex items-start gap-2">
        <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600 dark:text-red-400" aria-hidden="true" />
        <div className="min-w-0">
          <p className="font-semibold">Last preview run failed</p>
          <p className="mt-0.5 text-sm text-slate-800 dark:text-slate-100">{lastRunError}</p>
        </div>
      </div>
    </section>
  );

}
