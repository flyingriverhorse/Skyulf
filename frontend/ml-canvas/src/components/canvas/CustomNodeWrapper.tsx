import { memo, useEffect, useRef, useState } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from '@xyflow/react';
import { registry } from '../../core/registry/NodeRegistry';
import { AlertCircle, AlertTriangle, X, CheckCircle2, XCircle, Merge, GitFork } from 'lucide-react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useJobStore } from '../../core/store/useJobStore';
import { useViewStore } from '../../core/store/useViewStore';
import { bucketDuration, getPerfFamily } from '../../core/perf/perfThresholds';
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
  const { deleteElements, getEdges } = useReactFlow();
  // In read-only mode the per-node X is hidden along with the global
  // editor affordances (Backspace, sidebars, undo/redo, palette).
  const readOnly = useReadOnlyMode();

  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult: NodeExecutionResult | undefined = executionResult?.node_results?.[id];
  // Fallback summary entries for trainer/tuner nodes whose jobs run
  // via Celery — they never populate `executionResult.node_results`.
  // The `useNodeJobSummaries` hook keeps this map fresh from
  // `/jobs/node-summaries` after every job event. For parallel runs
  // the array contains one entry per branch; merge runs have one.
  const jobSummaries = useGraphStore((state) => state.nodeJobSummaries[id]);
  // Mirror of the canvas branch labels so multi-branch trainer cards
  // can show "Path B · Xgboost" letters that match the colored edges.
  const branchEdgeLabels = useGraphStore((state) => state.branchEdgeLabels);
  // Whether a fresh job is currently in flight for this node — lets
  // the card tag the existing (now-stale) summary as "previous run"
  // in its tooltip until the new job completes and overwrites it.
  const isJobInFlight = useJobStore((state) =>
    state.jobs.some(
      (j) => j.node_id === id && (j.status === 'running' || j.status === 'queued'),
    ),
  );
  const incomingSourceCount = useGraphStore(
    (state) => new Set(state.edges.filter((e) => e.target === id).map((e) => e.source)).size,
    (a, b) => a === b
  );

  // C7: pre-execution schema prediction.
  // - `predictedSchema` powers the `↳ N cols` badge in the header.
  //   `null` means data-dependent / unknown — we hide the badge.
  // - `brokenRefs` flags column references in this node's `params`
  //   that don't exist in the upstream's predicted schema (typo,
  //   deleted feature, etc.). Triggers a red border + alert chip.
  const predictedSchema = useGraphStore((state) => state.predictedSchemas[id]);
  const brokenRefs = useGraphStore((state) => state.brokenSchemaRefs[id]);
  const hasBrokenRefs = (brokenRefs?.length ?? 0) > 0;
  const brokenRefTooltip = hasBrokenRefs
    ? `⚠ Column name not found in upstream output\n` +
      (brokenRefs ?? [])
        .map((r) => `  • "${r.column}" (in field '${r.field}')`)
        .join('\n') +
      `\n\nThe canvas automatically previews each node's output schema in the background.\n` +
      `These column names were not found in the predicted upstream output — they may be\n` +
      `misspelled or the upstream step may have renamed / dropped them.\n\n` +
      `The pipeline can still run, but may fail at this step.`
    : null;
  // predictedSchema is `null` (not undefined) when the server explicitly
  // returned null — meaning this calculator is data-dependent (e.g. encoders,
  // feature-selection). `undefined` means the API hasn't responded yet.
  const schemaIsDataDependent = predictedSchema === null && definitionType !== 'dataset_node';

  // L4 perf overlay. When the user toggles the Toolbar gauge, every
  // card whose last run has a known wall-clock duration grows a
  // colored ring and a tooltip with exact ms.
  // Two duration sources:
  //   1. Preview-run path: `nodeResult.execution_time` (seconds).
  //   2. Trainer/tuner Celery path: latest `jobSummaries[i].duration_ms`.
  // Thresholds are family-aware (see `core/perf/perfThresholds.ts`):
  // preprocessing nodes finish in milliseconds, single-fit trainers in
  // seconds, and HPO/CV tuners legitimately run for minutes. Using one
  // flat threshold would paint every tuner red and convey no signal.
  const perfOverlayEnabled = useViewStore((s) => s.perfOverlayEnabled);
  const perfDurationMs: number | null = (() => {
    if (!perfOverlayEnabled) return null;
    if (typeof nodeResult?.execution_time === 'number') {
      return Math.max(0, Math.round(nodeResult.execution_time * 1000));
    }
    const fromJob = jobSummaries?.find((e) => typeof e.duration_ms === 'number');
    if (fromJob && typeof fromJob.duration_ms === 'number') return fromJob.duration_ms;
    return null;
  })();
  const perfBucket: 'fast' | 'medium' | 'slow' | null =
    perfDurationMs === null
      ? null
      : bucketDuration(perfDurationMs, getPerfFamily(definitionType));
  const perfRingClass =
    perfBucket === 'fast'
      ? 'ring-2 ring-green-500/60 ring-offset-1 ring-offset-background'
      : perfBucket === 'medium'
      ? 'ring-2 ring-amber-500/70 ring-offset-1 ring-offset-background'
      : perfBucket === 'slow'
      ? 'ring-2 ring-red-500/70 ring-offset-1 ring-offset-background'
      : '';
  const perfTelemetry = (() => {
    if (perfDurationMs === null) return null;
    
    // Core wall-clock duration message:
    const durStr =
      perfDurationMs >= 1000
        ? `${(perfDurationMs / 1000).toFixed(2)}s`
        : `${perfDurationMs}ms`;

    let fitStr: string | null = null;
    let memMB: number | null = null;
    let rowsStr: string | null = null;

    // Append granular metric details if Python sent them over:
    const m = nodeResult?.metrics;
    if (m) {
      if (typeof m.fit_time === 'number') {
        fitStr = m.fit_time >= 1 
          ? `${m.fit_time.toFixed(2)}s` 
          : `${Math.round(m.fit_time * 1000)}ms`;
      }
      if (typeof m.peak_memory_bytes === 'number') {
        memMB = m.peak_memory_bytes / (1024 * 1024);
      }
      if (typeof m.rows_in === 'number' && typeof m.rows_out === 'number') {
        rowsStr = `${m.rows_in} \u2192 ${m.rows_out}`;
      }
    }
    
    let tooltip = `Last run: ${durStr}`;
    if (fitStr) tooltip += `\nFit time: ${fitStr}`;
    if (memMB !== null) tooltip += `\nPeak mem: ${memMB.toFixed(1)} MB`;
    if (rowsStr) tooltip += `\nRows: ${rowsStr}`;

    return { durStr, fitStr, memMB, rowsStr, tooltip };
  })();

  const perfTooltip = perfTelemetry?.tooltip;

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
    <div
      data-testid={`canvas-node-${definitionType}`}
      data-node-definition-type={definitionType}
      data-perf-bucket={perfBucket ?? undefined}
      data-perf-duration-ms={perfDurationMs ?? undefined}
      title={perfTooltip}
      className={`
      relative group min-w-[200px] bg-card border-2 rounded-lg shadow-sm transition-all duration-150
      ${selected
        ? 'border-primary shadow-lg shadow-primary/30 scale-[1.02]'
        : validationMessage
        ? 'border-red-500/40 hover:border-red-500/60'
        : hasBrokenRefs
        ? 'border-amber-500/40 hover:border-amber-500/60'
        : 'border-border hover:border-primary/50'}
      ${isPulsing ? 'animate-validation-pulse' : ''}
      ${perfRingClass}
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
      {(nodeResult || validationMessage || hasBrokenRefs) && (
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
          {hasBrokenRefs && !validationMessage && (
            <span
              title={brokenRefTooltip ?? undefined}
              aria-label={`Column mismatch: ${brokenRefs?.length ?? 0} column${(brokenRefs?.length ?? 0) > 1 ? 's' : ''} not found in upstream output`}
              className="flex items-center justify-center w-5 h-5 rounded-full bg-amber-50 text-amber-600 border border-amber-200 shadow-sm dark:bg-amber-900/40 dark:text-amber-400 dark:border-amber-800"
            >
              <AlertTriangle size={11} />
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
            {predictedSchema && (
              <span
                className="shrink-0 flex items-center text-[10px] px-1.5 py-0.5 rounded-full font-medium bg-muted/60 text-muted-foreground border border-border/60"
                title={
                  `Canvas schema preview — predicted output before running:\n` +
                  `${predictedSchema.columns.length} column${predictedSchema.columns.length === 1 ? '' : 's'}:\n` +
                  predictedSchema.columns.slice(0, 12).join(', ') +
                  (predictedSchema.columns.length > 12 ? `, … (+${predictedSchema.columns.length - 12} more)` : '')
                }
              >
                ↳ {predictedSchema.columns.length} col{predictedSchema.columns.length === 1 ? '' : 's'}
              </span>
            )}
            {schemaIsDataDependent && (
              <span
                className="shrink-0 flex items-center text-[10px] px-1.5 py-0.5 rounded-full font-medium bg-muted/30 text-muted-foreground/60 border border-dashed border-border/50"
                title="Schema depends on data — this step's output columns can only be known after running the pipeline (e.g. one-hot encoding adds one column per category). Everything is fine; run the pipeline to see the actual output."
              >
                ↳ ?
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
          5. Nothing — collapse padding so card visually shrinks.
          Side padding is bumped on text bodies (px-10) so the body
          text never collides with the absolutely-positioned port
          labels ("Data" / "X" / "y" / "Train" / "Test") that float at
          left-4 / right-4. min-h gives the floating port labels
          vertical separation from the centered body line. */}
      {(() => {
        if (definition.component) {
          return (
            <div className="p-3">
              <definition.component data={data} />
            </div>
          );
        }
        // Priority chain for the body line:
        //  1. Inline preview run (`nodeResult.metadata.summary`) wins
        //     because it's always the freshest source for non-trainer
        //     nodes that run through `/preview`.
        //  2. Trainer/tuner Celery jobs land in `jobSummaries` instead.
        //     For parallel terminals this is an array (one entry per
        //     branch); for merge terminals it's a single entry.
        const inlineSummary = nodeResult?.metadata?.summary?.trim();
        const jobEntries = (jobSummaries ?? []).filter((e) => e.summary && e.summary.trim());
        if (inlineSummary || jobEntries.length > 0) {
          // Tooltip phrasing: when a fresh job is in flight, mark the
          // currently-rendered summary as the previous run so the user
          // knows the card hasn't updated yet.
          const tooltipPrefix = isJobInFlight ? 'Previous run · new run in progress—\n' : '';
          if (inlineSummary || jobEntries.length === 1) {
            const text = inlineSummary || jobEntries[0]!.summary;
            return (
              <div className="px-10 py-2 min-h-[2.75rem] flex items-center justify-center">
                <div
                  className="text-[11px] text-foreground/80 font-mono tabular-nums truncate text-center w-full"
                  title={`${tooltipPrefix}${text}`}
                >
                  {text}
                </div>
              </div>
            );
          }
          // Multi-branch: render one row per branch. Path labels come
          // from the canvas's `useBranchColors` map (mirrored into the
          // store as `branchEdgeLabels`) so the letters here always
          // match the colored "Path B · Xgboost" tags on the incoming
          // edges. We pair entries to incoming edges by order — both
          // backend (`partition_parallel_pipeline`) and the canvas
          // iterate `term.inputs` / `terminalIncoming` in the same
          // order, and `jobEntries` is already sorted by branch_index.
          const incomingEdges = getEdges()
            .filter((e) => e.target === id)
            .sort((a, b) => (a.sourceHandle ?? '').localeCompare(b.sourceHandle ?? ''));
          const rows = jobEntries.map((entry, idx) => {
            const edge = incomingEdges[idx];
            const fullLabel = edge ? branchEdgeLabels[edge.id] : undefined;
            // `Path X · Suffix` -> just `X` for the inline pill; full
            // label still goes into the tooltip below for context.
            let letter: string;
            const m = fullLabel?.match(/^Path\s+([A-Z])/);
            if (m) {
              letter = m[1]!;
            } else {
              const fallbackIdx = entry.branch_index ?? idx;
              letter = String.fromCharCode(65 + Math.max(0, fallbackIdx));
            }
            return {
              key: entry.pipeline_id,
              letter,
              tooltipLabel: fullLabel ?? `Path ${letter}`,
              summary: entry.summary,
            };
          });
          const tooltipBody = rows
            .map((r) => `${r.tooltipLabel}: ${r.summary}`)
            .join('\n');
          return (
            <div
              className="px-3 py-2 flex flex-col gap-0.5"
              title={`${tooltipPrefix}${tooltipBody}`}
            >
              {rows.map((r) => (
                <div
                  key={r.key}
                  className="text-[11px] text-foreground/80 font-mono tabular-nums truncate flex items-center gap-1.5"
                >
                  <span className="shrink-0 px-1 rounded bg-muted text-[10px] text-muted-foreground">
                    {r.letter}
                  </span>
                  <span className="truncate">{r.summary}</span>
                </div>
              ))}
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
            <div className="px-10 py-2 min-h-[2.75rem] flex items-center justify-center">
              <div
                className="text-[11px] text-muted-foreground truncate text-center w-full"
                title={preview}
              >
                {preview}
              </div>
            </div>
          );
        }
        if (definition.description) {
          return (
            <div className="px-10 py-2 min-h-[2.75rem] flex items-center justify-center">
              <div
                className="text-[11px] text-muted-foreground italic line-clamp-2 text-center w-full"
                title={definition.description}
              >
                {definition.description}
              </div>
            </div>
          );
        }
        return <div className="min-h-[1.5rem]" />;
      })()}

      {perfOverlayEnabled && perfTelemetry && (
        <div className="flex flex-row items-center justify-between gap-3 px-3 py-1.5 border-t border-border bg-muted/30 text-[9px] text-muted-foreground font-mono rounded-b-lg">
          <div className="flex flex-row items-center gap-2 min-w-0">
            <span className="truncate" title="Wall-clock time">⏱ {perfTelemetry.durStr}</span>
            {perfTelemetry.fitStr && <span className="truncate" title="Core fit time">⚡ {perfTelemetry.fitStr}</span>}
          </div>
          {perfTelemetry.memMB !== null && <span className="shrink-0 font-semibold" title="Peak Memory">💾 {perfTelemetry.memMB.toFixed(1)}MB</span>}
        </div>
      )}

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
