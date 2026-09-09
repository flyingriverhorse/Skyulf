import { ChevronUp, ChevronDown, Maximize2, Minimize2, Table, X } from 'lucide-react';
import type { PreviewResponse } from '../../../../core/api/client';

interface PanelActionsProps {
  isResultsPanelExpanded: boolean;
  setResultsPanelExpanded: (expanded: boolean) => void;
  isMaximized: boolean;
  setIsMaximized: (maximized: boolean) => void;
  canClose: boolean;
  onClose: () => void;
}

interface PanelHeaderProps extends Omit<PanelActionsProps, 'canClose'> {
  executionResult: PreviewResponse | null;
  lastRunError: string | null;
  validationCount: number;
  previewCount: number;
  currentTotal: number;
  branchCount: number;
  issueCount: number;
}

/** Show preview totals alongside persistent panel controls. */
export function PanelHeader({
  executionResult,
  lastRunError,
  validationCount,
  previewCount,
  currentTotal,
  branchCount,
  issueCount,
  isResultsPanelExpanded,
  setResultsPanelExpanded,
  isMaximized,
  setIsMaximized,
  onClose,
}: PanelHeaderProps) {
  return (
    <div
      className="flex items-center justify-between gap-2 px-4 py-2 bg-muted/10 border-b select-none shrink-0"
    >
      <button
        type="button"
        aria-label="Toggle preview results"
        aria-expanded={isResultsPanelExpanded}
        onClick={() => setResultsPanelExpanded(!isResultsPanelExpanded)}
        className="flex flex-1 items-center gap-2 min-w-0 text-left text-foreground rounded hover:bg-muted/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      >
        <Table className="w-4 h-4 text-primary shrink-0" />
        <span className="font-semibold text-sm truncate">Preview Results</span>
        {executionResult && (
          <span className="text-xs text-muted-foreground truncate">
            {previewCount === currentTotal
              ? `${currentTotal} rows`
              : `${previewCount} of ${currentTotal} rows shown`}
            {branchCount > 0 ? ` · ${branchCount} branches` : ''}
          </span>
        )}
        {executionResult?.status === 'failed' && (
          <span className="text-xs text-red-600 font-bold shrink-0">(Failed)</span>
        )}
        <IssueCountBadge issueCount={issueCount} />
      </button>
      <PanelActions
        isResultsPanelExpanded={isResultsPanelExpanded}
        setResultsPanelExpanded={setResultsPanelExpanded}
        isMaximized={isMaximized}
        setIsMaximized={setIsMaximized}
        canClose={Boolean(executionResult || lastRunError || validationCount > 0)}
        onClose={onClose}
      />
    </div>
  );
}

/** Keep collapse, maximize and dismissal actions independent of the active pane. */
function PanelActions({
  isResultsPanelExpanded,
  setResultsPanelExpanded,
  isMaximized,
  setIsMaximized,
  canClose,
  onClose,
}: PanelActionsProps) {
  return (
    <div className="flex items-center gap-1 shrink-0">
      {isResultsPanelExpanded && (
        <button
          type="button"
          className="p-1 hover:bg-muted rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          aria-label={isMaximized ? 'Restore results panel' : 'Maximize results panel'}
          onClick={(e) => {
            e.stopPropagation();
            setIsMaximized(!isMaximized);
          }}
        >
          {isMaximized ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </button>
      )}
      <button
        type="button"
        className="p-1 hover:bg-muted rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        aria-label={isResultsPanelExpanded ? 'Collapse results panel' : 'Expand results panel'}
        aria-expanded={isResultsPanelExpanded}
        onClick={() => setResultsPanelExpanded(!isResultsPanelExpanded)}
      >
        {isResultsPanelExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
      </button>
      {canClose && (
        <button
          type="button"
          className="p-1 hover:bg-muted rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          aria-label="Close preview results"
          title="Close preview results"
          onClick={(e) => {
            e.stopPropagation();
            onClose();
          }}
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}

function IssueCountBadge({ issueCount }: { issueCount: number }) {
  return issueCount > 0 && (
    <span className="shrink-0 text-[11px] font-medium px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300 whitespace-nowrap">
      {issueCount} {issueCount === 1 ? 'issue' : 'issues'}
    </span>
  );
}
