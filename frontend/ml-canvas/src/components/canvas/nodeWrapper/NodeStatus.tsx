import type { NodeProps } from '@xyflow/react';
import type { NodeDefinition } from '../../../core/types/nodes';
import type { NodeExecutionResult } from '../../../core/api/client';
import type { NodePresentation } from './useNodePresentation';
import type { NodePerformance } from './nodePerformance';
import { AlertCircle, AlertTriangle, X, CheckCircle2, XCircle } from 'lucide-react';
import { LeakageIssuePopover } from '../LeakageIssuePopover';

export function UnknownNode({ definitionType }: { definitionType: string }) {
  return (
    <div className="p-4 border-2 border-destructive bg-destructive/10 rounded-md min-w-[150px]">
      <div className="flex items-center gap-2 text-destructive">
        <AlertCircle size={16} />
        <span className="text-sm font-bold">Unknown Node</span>
      </div>
      <div className="text-xs mt-1">Type: {definitionType}</div>
    </div>
  );
}

export function NodeDeleteButton({ readOnly, selected, onDelete }: {
  readOnly: boolean; selected: NodeProps['selected']; onDelete: (event: React.MouseEvent) => void;
}) {
  return (<>
    {!readOnly && (
      <button
        onClick={onDelete}
        aria-label="Remove node"
        title="Remove node"
        className={`absolute -top-2 -right-2 z-10 flex items-center justify-center w-6 h-6 rounded-full bg-background border border-border shadow-sm text-muted-foreground hover:bg-destructive hover:text-destructive-foreground hover:border-destructive transition-all ${selected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100 focus-visible:opacity-100'
          }`}
      >
        <X size={12} />
      </button>
    )}
  </>);
}

function RunStatus({ nodeResult }: { nodeResult: NodeExecutionResult | undefined }) {
  return (<>
    {nodeResult && (
      <span
        title={nodeResult.status === 'success'
          ? 'Last run: success'
          : `Last run failed: Check Notifications or Error Page for details.`
        }
        aria-label={nodeResult.status === 'success' ? 'Last run: success' : 'Last run: failed'}
        className={`flex items-center justify-center w-5 h-5 rounded-full border shadow-sm ${nodeResult.status === 'success'
            ? 'bg-green-50 text-green-700 border-green-200 dark:bg-green-900/40 dark:text-green-400 dark:border-green-900'
            : 'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/40 dark:text-red-400 dark:border-red-900'
          }`}
      >
        {nodeResult.status === 'success' ? <CheckCircle2 size={11} /> : <XCircle size={11} />}
      </span>
    )}
  </>);
}

function ConfigurationStatus({ schema, validation }: Pick<NodePresentation, 'schema' | 'validation'>) {
  const { validationMessage } = validation;
  const { hasBrokenRefs, brokenRefs, brokenRefTooltip } = schema;
  if (validationMessage) return (
    <span
      title={validationMessage}
      aria-label={`Configuration issue: ${validationMessage}`}
      className="flex items-center justify-center w-5 h-5 rounded-full bg-red-50 text-red-600 border border-red-200 shadow-sm dark:bg-red-900/40 dark:text-red-400 dark:border-red-900"
    >
      <AlertCircle size={11} />
    </span>
  );
  if (!hasBrokenRefs) return null;
  const count = brokenRefs?.length ?? 0;
  return (
    <span
      title={brokenRefTooltip ?? undefined}
      aria-label={`Column mismatch: ${count} column${count > 1 ? 's' : ''} not found in upstream output`}
      className="flex items-center justify-center w-5 h-5 rounded-full bg-amber-50 text-amber-600 border border-amber-200 shadow-sm dark:bg-amber-900/40 dark:text-amber-400 dark:border-amber-800"
    >
      <AlertTriangle size={11} />
    </span>
  );
}

export function NodeStatus({ definition, leakage, execution, schema, validation }:
  Pick<NodePresentation, 'leakage' | 'execution' | 'schema' | 'validation'> & { definition: NodeDefinition<unknown> }) {
  const { leakageSeverity, leakageIssues, openGuide } = leakage;
  const { nodeResult } = execution;
  const { validationMessage } = validation;
  const { hasBrokenRefs } = schema;
  return (<>
    {(leakageSeverity || nodeResult || validationMessage || hasBrokenRefs) && (
      <div className="absolute -top-2 -left-2 z-10 flex items-center gap-1">
        {leakageSeverity && <LeakageIssuePopover issues={leakageIssues} subject={definition.label} openGuide={openGuide} compact />}
        <RunStatus nodeResult={nodeResult} />
        <ConfigurationStatus schema={schema} validation={validation} />
      </div>
    )}
  </>);
}

export function NodePerformanceFooter({ perf }: { perf: NodePerformance }) {
  const { perfOverlayEnabled, perfTelemetry } = perf;
  return (<>
    {perfOverlayEnabled && perfTelemetry && (
      <div className="flex flex-row items-center justify-between gap-3 px-3 py-1.5 border-t border-border bg-muted/30 text-[9px] text-muted-foreground font-mono rounded-b-lg">
        <div className="flex flex-row items-center gap-2 min-w-0">
          <span className="truncate" title="Wall-clock time">⏱ {perfTelemetry.durStr}</span>
          {perfTelemetry.fitStr && <span className="truncate" title="Core fit time">⚡ {perfTelemetry.fitStr}</span>}
        </div>
        {perfTelemetry.memMB !== null && <span className="shrink-0 font-semibold" title="Peak Memory">💾 {perfTelemetry.memMB.toFixed(1)}MB</span>}
      </div>
    )}
  </>);
}
