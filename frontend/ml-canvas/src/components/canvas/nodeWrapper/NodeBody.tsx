import type { Edge } from '@xyflow/react';
import type { NodeDefinition } from '../../../core/types/nodes';
import type { NodeSummaryEntry } from '../../../core/api/jobs';
import type { NodePresentation } from './useNodePresentation';

const splitBodyPadding = 'pl-3 pr-[var(--split-output-space,0px)]';
function getBodyTextClass(hasMultipleOutputs: boolean): string {
  return `${hasMultipleOutputs ? splitBodyPadding : 'px-10'} py-2 min-h-[2.75rem] flex items-center justify-center`;
}

interface BodyProps {
  id: string;
  data: Record<string, unknown>;
  definition: NodeDefinition<unknown>;
  execution: NodePresentation['execution'];
  hasMultipleOutputs: boolean;
  getEdges: () => Edge[];
}

function NodeBranchSummaries({ id, getEdges, execution, hasMultipleOutputs, jobEntries, tooltipPrefix }:
  Pick<BodyProps, 'id' | 'getEdges' | 'execution' | 'hasMultipleOutputs'> & {
    jobEntries: NodeSummaryEntry[]; tooltipPrefix: string;
  }) {
  const { branchEdgeLabels } = execution;
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
      className={`${hasMultipleOutputs ? splitBodyPadding : 'px-3'} py-2 flex flex-col gap-0.5`}
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

function NodeSummaries({ id, getEdges, execution, hasMultipleOutputs, jobEntries, inlineSummary, bodyTextClass }:
  Pick<BodyProps, 'id' | 'getEdges' | 'execution' | 'hasMultipleOutputs'> & {
    jobEntries: NodeSummaryEntry[]; inlineSummary: string | undefined; bodyTextClass: string;
  }) {
  const { isJobInFlight } = execution;
  // Tooltip phrasing: when a fresh job is in flight, mark the
  // currently-rendered summary as the previous run so the user
  // knows the card hasn't updated yet.
  const tooltipPrefix = isJobInFlight ? 'Previous run · new run in progress—\n' : '';
  if (inlineSummary || jobEntries.length === 1) {
    const text = inlineSummary || jobEntries[0]!.summary;
    return (
      <div className={bodyTextClass}>
        <div
          className="text-[11px] text-foreground/80 font-mono tabular-nums truncate text-center w-full"
          title={`${tooltipPrefix}${text}`}
        >
          {text}
        </div>
      </div>
    );
  }
  return <NodeBranchSummaries id={id} getEdges={getEdges} execution={execution}
    hasMultipleOutputs={hasMultipleOutputs} jobEntries={jobEntries} tooltipPrefix={tooltipPrefix} />;
}

function getConfigPreview(definition: NodeDefinition<unknown>, data: Record<string, unknown>) {
  let preview: string | null = null;
  if (definition.bodyPreview) {
    try {
      preview = definition.bodyPreview(data);
    } catch {
      // A buggy preview must not break the canvas.
      preview = null;
    }
  }
  return preview;
}

function NodeConfigPreview({ definition, data, bodyTextClass }: Pick<BodyProps, 'definition' | 'data'>
  & { bodyTextClass: string }) {
  const preview = getConfigPreview(definition, data);
  if (preview && preview.trim()) {
    return (
      <div className={bodyTextClass}>
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
      <div className={bodyTextClass}>
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
}

/** Custom views win, followed by inline results, job summaries, config preview, and description. */
export function NodeBody({ id, data, definition, execution, hasMultipleOutputs, getEdges }: BodyProps) {
  const { nodeResult, jobSummaries } = execution;
  const bodyTextClass = getBodyTextClass(hasMultipleOutputs);
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
    return <NodeSummaries id={id} getEdges={getEdges} execution={execution}
      hasMultipleOutputs={hasMultipleOutputs} jobEntries={jobEntries} inlineSummary={inlineSummary}
      bodyTextClass={bodyTextClass} />;
  }
  return <NodeConfigPreview definition={definition} data={data} bodyTextClass={bodyTextClass} />;
}
