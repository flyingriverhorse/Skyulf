import { memo, useEffect, useRef, useState } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from '@xyflow/react';
import { registry } from '../../core/registry/NodeRegistry';
import { AlertCircle, X, CheckCircle2, XCircle, Merge, GitFork } from 'lucide-react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useReadOnlyMode } from '../../core/hooks/useReadOnlyMode';
import type { NodeExecutionResult } from '../../core/api/client';
import {
  isAutoParallelType,
  supportsExecutionModeToggle,
  getExecutionMode,
} from '../../core/types/executionMode';

function CustomNodeWrapperImpl({ id, data, selected }: NodeProps) {
  const definitionType = data.definitionType as string;
  const definition = registry.get(definitionType);
  const { deleteElements } = useReactFlow();
  // In read-only mode the per-node X is hidden along with the global
  // editor affordances (Backspace, sidebars, undo/redo, palette).
  const readOnly = useReadOnlyMode();

  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult: NodeExecutionResult | undefined = executionResult?.node_results?.[id];
  const incomingSourceCount = useGraphStore(
    (state) => new Set(state.edges.filter((e) => e.target === id).map((e) => e.source)).size,
    (a, b) => a === b
  );

  // Has this node received an active sibling-fan-in advisory with real
  // overlap (i.e. last-wins overwrite is happening)? If so, color the
  // merge badge amber so the canvas itself flags the risk.
  const mergeWarningSeverity: 'risk' | 'safe' | null = (() => {
    const warnings = executionResult?.merge_warnings ?? [];
    const w = warnings.find((mw) => mw.node_id === id);
    if (!w) return null;
    return (w.overlap_columns?.length ?? 0) > 0 ? 'risk' : 'safe';
  })();

  // Parallel badge: training nodes when user explicitly chose parallel mode,
  // OR auto-parallel terminals (data_preview) wired to 2+ sources.
  const isTrainingNode = supportsExecutionModeToggle(definitionType);
  const isAutoParallel = isAutoParallelType(definitionType);
  const isParallel =
    (isTrainingNode && getExecutionMode(data) === 'parallel') ||
    (isAutoParallel && incomingSourceCount > 1);

  // Only nodes that actually consume upstream data (i.e. declare an input
  // port) can merge. Dataset/data-loader nodes have no inputs and must not
  // show a merge badge even if React Flow allowed an edge in. Auto-parallel
  // terminals show the parallel badge instead of the merge badge.
  const canMerge = (definition?.inputs?.length ?? 0) > 0 && !isAutoParallel;
  const showMergeBadge = (canMerge && incomingSourceCount > 1) || isParallel;

  // Inline validation: surface a small red dot in the header when the
  // node's own `validate(config)` returns invalid, so users see at a
  // glance which nodes still need configuration. The validator runs
  // against the node's data (data carries the user config fields plus
  // some metadata; validators only read the config keys). Wrapped in
  // try/catch because a buggy validator must not crash the canvas.
  let validationMessage: string | null = null;
  if (definition) {
    try {
      const result = definition.validate(data as never);
      if (!result.isValid) {
        validationMessage = result.message ?? 'Configuration incomplete.';
      }
    } catch {
      // Validator threw — treat as a soft warning, don't block rendering.
      validationMessage = null;
    }
  }

  // One-shot pulse animation: when a node transitions from valid →
  // invalid, play a 5 s red-ring pulse to draw attention. `isPulsing`
  // gates the CSS class; we only set it on the false→true edge of
  // `isInvalid`, never every render, so an already-invalid node
  // doesn't re-pulse forever. The class auto-clears after 5 s.
  const wasInvalidRef = useRef<boolean>(false);
  const [isPulsing, setIsPulsing] = useState<boolean>(false);
  const isInvalid = validationMessage !== null;
  useEffect(() => {
    if (isInvalid && !wasInvalidRef.current) {
      setIsPulsing(true);
      // Matches the CSS animation: 3 cycles × 1.6 s = 4.8 s.
      const t = window.setTimeout(() => setIsPulsing(false), 4800);
      wasInvalidRef.current = true;
      return () => window.clearTimeout(t);
    }
    if (!isInvalid && wasInvalidRef.current) {
      // Cleared by the user fixing config — drop the pulse early.
      wasInvalidRef.current = false;
      setIsPulsing(false);
    }
    return undefined;
  }, [isInvalid]);

  const onDelete = (evt: React.MouseEvent) => {
    evt.stopPropagation();
    deleteElements({ nodes: [{ id }] });
  };

  if (!definition) {
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

  // Determine handle positions based on port definitions
  // This is a simplified version. In a real app, we might want more control over handle placement.
  
  return (
    <div className={`
      relative group min-w-[200px] bg-card border-2 rounded-lg shadow-sm transition-all duration-150
      ${selected
        ? 'border-primary shadow-lg shadow-primary/30 scale-[1.02]'
        : validationMessage
        ? 'border-red-500/40 hover:border-red-500/60'
        : 'border-border hover:border-primary/50'}
      ${isPulsing ? 'animate-validation-pulse' : ''}
    `}>
      {/* Floating delete chip — absolute on the card corner so it never
          competes with header text/badges for flex space. Hidden in
          read-only and revealed on hover or when selected. */}
      {!readOnly && (
        <button
          onClick={onDelete}
          aria-label="Remove node"
          title="Remove node"
          className={`absolute -top-2 -right-2 z-10 flex items-center justify-center w-6 h-6 rounded-full bg-background border border-border shadow-sm text-muted-foreground hover:bg-destructive hover:text-destructive-foreground hover:border-destructive transition-all ${
            selected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100 focus-visible:opacity-100'
          }`}
        >
          <X size={12} />
        </button>
      )}
      {/* Floating status/validation chips at the top-left corner. Kept
          out of the header text row so they can't be squeezed by long
          titles and out of the right edge so they can't collide with
          output-handle labels (which absolute-position at 25/50/75%
          of the card height for multi-output nodes like splitters). */}
      {(nodeResult || validationMessage) && (
        <div className="absolute -top-2 -left-2 z-10 flex items-center gap-1">
          {nodeResult && (
            <span
              title={nodeResult.status === 'success' ? 'Last run: success' : 'Last run: failed'}
              aria-label={nodeResult.status === 'success' ? 'Last run: success' : 'Last run: failed'}
              className={`flex items-center justify-center w-5 h-5 rounded-full border shadow-sm ${
                nodeResult.status === 'success'
                  ? 'bg-green-50 text-green-700 border-green-200 dark:bg-green-900/40 dark:text-green-400 dark:border-green-900'
                  : 'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/40 dark:text-red-400 dark:border-red-900'
              }`}
            >
              {nodeResult.status === 'success' ? <CheckCircle2 size={11} /> : <XCircle size={11} />}
            </span>
          )}
          {validationMessage && (
            <span
              title={validationMessage}
              aria-label={`Configuration issue: ${validationMessage}`}
              className="flex items-center justify-center w-5 h-5 rounded-full bg-red-50 text-red-600 border border-red-200 shadow-sm dark:bg-red-900/40 dark:text-red-400 dark:border-red-900"
            >
              <AlertCircle size={11} />
            </span>
          )}
        </div>
      )}
      {/* Header */}
      <div className="flex items-center p-3 border-b bg-muted/30 rounded-t-lg">
        <div className="p-1.5 bg-primary/10 rounded mr-3 shrink-0">
          {definition.icon && <definition.icon className="w-4 h-4 text-primary" />}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <div className="text-sm font-bold truncate" title={definition.label}>{definition.label}</div>
            {showMergeBadge && (
              <span
                className={`shrink-0 flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
                  isParallel
                    ? 'bg-amber-500/15 text-amber-500'
                    : mergeWarningSeverity === 'risk'
                    ? 'bg-amber-500/20 text-amber-600 dark:text-amber-400 ring-1 ring-amber-500/40'
                    : 'bg-blue-500/15 text-blue-400'
                }`}
                title={
                  isParallel
                    ? `Parallel: ${incomingSourceCount} branches will run as separate experiments`
                    : mergeWarningSeverity === 'risk'
                    ? `Merge with overlap: ${incomingSourceCount} branches share columns — last input overwrites earlier ones. See Results panel banner for details.`
                    : `Merge: combining data from ${incomingSourceCount} upstream sources`
                }
              >
                {isParallel ? <GitFork size={10} /> : <Merge size={10} />}
                {incomingSourceCount}
              </span>
            )}
          </div>
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
            {definition.category}
          </div>
        </div>
      </div>

      {/* Body — priority chain (see temp/node_body_content_plan.md):
          1. Inspection-class nodes with a custom `component` always win.
          2. Backend post-run summary (`nodeResult.metadata.summary`).
          3. Frontend pre-run preview (`definition.bodyPreview(data)`).
          4. Static italic description.
          5. Nothing — collapse padding so card visually shrinks. */}
      {(() => {
        if (definition.component) {
          return (
            <div className="p-3">
              <definition.component data={data} />
            </div>
          );
        }
        const summary = nodeResult?.metadata?.summary?.trim();
        if (summary) {
          return (
            <div className="px-3 pt-1 pb-2">
              <div
                className="text-[11px] text-foreground/80 font-mono tabular-nums truncate"
                title={summary}
              >
                {summary}
              </div>
            </div>
          );
        }
        let preview: string | null = null;
        if (definition.bodyPreview) {
          try {
            preview = definition.bodyPreview(data as never);
          } catch {
            // A buggy preview must not break the canvas.
            preview = null;
          }
        }
        if (preview && preview.trim()) {
          return (
            <div className="px-3 pt-1 pb-2">
              <div
                className="text-[11px] text-muted-foreground truncate"
                title={preview}
              >
                {preview}
              </div>
            </div>
          );
        }
        if (definition.description) {
          return (
            <div className="px-3 pt-1 pb-2">
              <div
                className="text-[11px] text-muted-foreground italic line-clamp-2"
                title={definition.description}
              >
                {definition.description}
              </div>
            </div>
          );
        }
        return <div className="pb-1" />;
      })()}

      {/* Input Handles */}
      {definition.inputs.map((input, index) => (
        <Handle
          key={`input-${input.id}`}
          type="target"
          position={Position.Left}
          id={input.id}
          className="!w-3 !h-3 !bg-muted-foreground hover:!bg-primary transition-colors"
          style={{ top: `${((index + 1) * 100) / (definition.inputs.length + 1)}%` }}
        >
          <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground pointer-events-none whitespace-nowrap px-1 rounded bg-card/80 backdrop-blur-[1px]">
            {input.label}
          </div>
        </Handle>
      ))}

      {/* Output Handles */}
      {definition.outputs.map((output, index) => (
        <Handle
          key={`output-${output.id}`}
          type="source"
          position={Position.Right}
          id={output.id}
          className="!w-3 !h-3 !bg-muted-foreground hover:!bg-primary transition-colors"
          style={{ top: `${((index + 1) * 100) / (definition.outputs.length + 1)}%` }}
        >
          <div className="absolute right-4 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground pointer-events-none whitespace-nowrap px-1 rounded bg-card/80 backdrop-blur-[1px]">
            {output.label}
          </div>
        </Handle>
      ))}
    </div>
  );
}

export const CustomNodeWrapper = memo(CustomNodeWrapperImpl);
