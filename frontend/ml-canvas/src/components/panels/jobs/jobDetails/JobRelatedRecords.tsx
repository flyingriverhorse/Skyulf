import React, { useState, useMemo, useCallback, useId } from 'react';
import { Check, Copy, ChevronsDown, ChevronsUp, FlaskConical, GitBranch, Workflow, Layers, Info, Link2, ScanEye, Tag } from 'lucide-react';
import { JobInfo } from '../../../../core/api/jobs';
import { RecordLink, NodeInspectorLink } from '../../../shared';
import { JobRecordContext } from './types';

/** A job's pipeline_id parsed into its execution-run context. */
interface PipelineRunContext {
  /** Id of the run this branch was split from, or null when the id isn't a branch. */
  parentPipelineId: string | null;
  /** Zero-based branch index, or null when the id isn't a branch. */
  branchIndex: number | null;
  /** True when the run (or its parent) is a synthetic preview execution, not a saved pipeline. */
  isPreviewRun: boolean;
}

// Mirrors `backend/ml_pipeline/_execution/utils.py::parse_branch_info` so a
// branch suffix is recognised identically on both sides of the API boundary.
const BRANCH_ID_RE = /^(.+?)__branch_(\d+)(?:_(\d+))?$/;

/**
 * Parses a job's pipeline_id into its branch/preview run context.
 *
 * Never invents a navigable target: ids that don't match a recognised shape
 * are treated as an opaque single run rather than guessed at.
 */
function parsePipelineRunContext(pipelineId: string): PipelineRunContext {
  const match = BRANCH_ID_RE.exec(pipelineId);
  const parentPipelineId = match ? match[1]! : null;
  const branchIndex = match ? Number(match[2]) : null;
  const previewSource = parentPipelineId ?? pipelineId;
  return { parentPipelineId, branchIndex, isPreviewRun: previewSource.startsWith('preview_') };
}

/**
 * Expandable "Related" entry for a job's pipeline execution id.
 *
 * Saved pipeline configurations are only ever keyed by dataset id (see
 * `backend/ml_pipeline/_internal/_routers/pipelines_io.py`), so a job's
 * `pipeline_id` — including synthetic `preview_*` and `*__branch_N` runtime
 * ids — never resolves to a separately viewable saved pipeline. Rather than
 * rendering a link into that dead end, this reveals the actual run context
 * (branch/preview status, node count, full id) inline on activation.
 */
const RelatedPipelineEntry: React.FC<{ job: JobInfo }> = ({ job }) => {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const panelId = useId();
  const pipelineId = job.pipeline_id;
  const context = useMemo(() => parsePipelineRunContext(pipelineId), [pipelineId]);
  const nodeCount = (job.graph as { nodes?: unknown[] } | undefined)?.nodes?.length;

  const handleCopy = useCallback(() => {
    void navigator.clipboard.writeText(pipelineId).then(() => {
      setCopied(true);
      setTimeout(() => { setCopied(false); }, 1500);
    });
  }, [pipelineId]);

  const { summaryLabel, RunIcon, iconTint } = getRunAppearance(context);

  return (
    <div className="flex min-w-64 flex-1 flex-col gap-1.5">
      <button
        type="button"
        aria-expanded={expanded}
        aria-controls={panelId}
        onClick={() => { setExpanded(v => !v); }}
        title={pipelineId}
        className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border text-xs font-medium transition-colors self-start ${expanded
          ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300'
          : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-blue-600 dark:text-blue-400 hover:border-blue-300 dark:hover:border-blue-600 hover:shadow-sm'
          }`}
      >
        <RunIcon className={`w-3.5 h-3.5 ${iconTint}`} aria-hidden="true" />
        {summaryLabel}
        {expanded
          ? <ChevronsUp className="w-3 h-3 text-gray-400 dark:text-gray-500" aria-hidden="true" />
          : <ChevronsDown className="w-3 h-3 text-gray-400 dark:text-gray-500" aria-hidden="true" />}
      </button>
      {expanded && (
        <PipelineRunDetails panelId={panelId} pipelineId={pipelineId} context={context} nodeCount={nodeCount} copied={copied} handleCopy={handleCopy} />
      )}
    </div>
  );
};

export function JobRelatedRecords({ job, ...context }: JobRecordContext) {
  const hasPipelineRun = isPipelineRun(job);

  return (
    <>
      {hasPipelineRun || job.promoted_at ? (
        <div className="flex flex-wrap items-start gap-2 text-xs p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
          <span className="flex items-center gap-1.5 pt-1.5 text-gray-500 dark:text-gray-400 font-medium mr-1">
            <Link2 className="w-3.5 h-3.5" aria-hidden="true" />
            Related
          </span>
          {hasPipelineRun && (
            <RelatedPipelineEntry job={job} />
          )}
          <div className="ml-auto flex shrink-0 items-center gap-2">
            <JobNodeLink job={job} {...context} />
            <JobModelVersionLink job={job} {...context} />
          </div>
        </div>
      ) : null}
    </>
  );
}

function getRunAppearance(context: PipelineRunContext) {
  if (context.isPreviewRun) {
    return { summaryLabel: 'Preview run', RunIcon: FlaskConical, iconTint: 'text-purple-500 dark:text-purple-400' };
  }
  if (context.parentPipelineId) {
    return { summaryLabel: `Branch ${context.branchIndex}`, RunIcon: GitBranch, iconTint: 'text-amber-500 dark:text-amber-400' };
  }
  return { summaryLabel: 'Pipeline run', RunIcon: Workflow, iconTint: 'text-blue-500 dark:text-blue-400' };
}

function PipelineRunDetails({ panelId, pipelineId, context, nodeCount, copied, handleCopy }: { panelId: string; pipelineId: string; context: PipelineRunContext; nodeCount: number | undefined; copied: boolean; handleCopy: () => void }) {
  return (
    <div id={panelId} className="w-full p-3 space-y-2.5 text-xs bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
      <div>
        <div className="text-[10px] font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wider mb-1">Run id</div>
        <div className="flex items-start gap-1.5 p-2 rounded bg-gray-50 dark:bg-gray-900/50 border border-gray-100 dark:border-gray-700/60">
          <span className="font-mono break-all text-gray-700 dark:text-gray-300">{pipelineId}</span>
          <button
            type="button"
            onClick={handleCopy}
            aria-label={copied ? 'Run id copied' : 'Copy run id'}
            className="shrink-0 rounded p-0.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          >
            {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
          </button>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {context.parentPipelineId && (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 text-purple-700 dark:text-purple-300">
            <GitBranch className="w-3 h-3 shrink-0" aria-hidden="true" />
            Parent run: <span className="font-mono break-all">{context.parentPipelineId}</span> (branch {context.branchIndex})
          </span>
        )}
        {nodeCount !== undefined && (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-gray-50 dark:bg-gray-900/40 border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300">
            <Layers className="w-3 h-3 shrink-0 text-gray-400 dark:text-gray-500" aria-hidden="true" />
            Nodes executed: {nodeCount}
          </span>
        )}
      </div>
      <div className="flex items-start gap-1.5 p-2 rounded bg-blue-50/60 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-900/40 text-gray-600 dark:text-gray-400">
        <Info className="w-3 h-3 mt-0.5 shrink-0 text-blue-400 dark:text-blue-500" aria-hidden="true" />
        <span className="italic">
          {context.isPreviewRun
            ? "Preview run — not a saved pipeline. This id only identifies this job's execution and can't be reopened in the pipeline canvas."
            : "This is the execution run id for this job, not a separately saved pipeline. Use the Dataset link above to reopen the pipeline that produced it."}
        </span>
      </div>
    </div>
  );
}

function JobNodeLink({ job, origin, filters }: JobRecordContext) {
  return (
    <>
      {isPipelineRun(job) && job.node_id && (
        <NodeInspectorLink
          nodeId={job.node_id}
          jobId={job.job_id}
          pipelineId={job.pipeline_id}
          label={(
            <>
              <ScanEye className="w-3.5 h-3.5 text-blue-500 dark:text-blue-400" aria-hidden="true" />
              Inspect node
            </>
          )}
          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 font-medium no-underline hover:border-blue-300 dark:hover:border-blue-600 hover:shadow-sm transition-colors"
          {...(origin !== undefined ? { origin } : {})}
          {...(filters !== undefined ? { filters } : {})}
        />
      )}
    </>
  );
}

function JobModelVersionLink({ job, origin, filters }: JobRecordContext) {
  return (
    <>
      {job.promoted_at && job.version !== undefined && (
        <RecordLink
          recordRef={{ kind: 'modelVersion', jobId: job.job_id, version: String(job.version) }}
          label={(
            <>
              <Tag className="w-3.5 h-3.5 text-amber-500 dark:text-amber-400" aria-hidden="true" />
              Model version {job.version}
            </>
          )}
          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 font-medium no-underline hover:border-blue-300 dark:hover:border-blue-600 hover:shadow-sm transition-colors"
          {...(origin !== undefined ? { origin } : {})}
          {...(filters !== undefined ? { filters } : {})}
        />
      )}
    </>
  );
}

function isPipelineRun(job: JobInfo): boolean {
  return !!job.pipeline_id && job.pipeline_id !== 'eda' && job.pipeline_id !== 'ingestion';
}
