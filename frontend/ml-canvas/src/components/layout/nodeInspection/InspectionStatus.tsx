import type { InspectionSide } from '../../../core/types/nodeInspection';
import type { InspectionDirection } from './useInspectionSelection';

/** Keep loading, stale-receipt and refresh-error feedback visible alongside captured rows. */
export function InspectionStatus({ isLoading, isStale, capturedSide, side, error }: {
  isLoading: boolean; isStale: boolean; capturedSide: InspectionSide | undefined;
  side: InspectionDirection; error: string | null;
}) {
  return <>
    <div role="status" aria-live="polite" className="space-y-2 text-xs">
      {isLoading && <p className="text-muted-foreground">Running preview…</p>}
      {isStale && <p className="rounded-md border border-amber-300 bg-amber-50 p-2 text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
        <strong>Stale preview.</strong> The graph changed after this capture. Run Preview data in the toolbar to measure the current settings.
      </p>}
      {!capturedSide && !isLoading && <p className="rounded-md border bg-muted/30 p-3 text-muted-foreground">
        No measured {side} yet. Run Preview data in the toolbar to inspect this node.
      </p>}
    </div>
    {error && <p role="alert" className="rounded-md border border-destructive/40 p-2 text-xs text-destructive">{error}</p>}
  </>;
}
