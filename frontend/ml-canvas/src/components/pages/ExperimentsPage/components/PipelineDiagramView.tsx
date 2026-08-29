// Renders the mermaid topology diagram each successful run persists on
// `metrics.pipeline_diagram` (built by backend/ml_pipeline/_execution/diagram.py).
// One card per selected run; runs without a diagram (legacy/pre-database rows)
// are listed as such instead of silently dropped.

import React from 'react';
import type { JobInfo } from '../../../../core/api/jobs';
import { shortRunId } from '../utils/jobMeta';
import { MermaidDiagram } from './MermaidDiagram';

interface Props {
  jobs: JobInfo[];
}

function getDiagram(job: JobInfo): string | null {
  const metrics = (job.metrics ?? job.result?.metrics ?? {}) as Record<string, unknown>;
  const diagram = metrics.pipeline_diagram;
  return typeof diagram === 'string' && diagram.length > 0 ? diagram : null;
}

const CopyMermaidButton: React.FC<{ diagram: string }> = ({ diagram }) => {
  const [state, setState] = React.useState<'idle' | 'copied' | 'error'>('idle');
  const timer = React.useRef<number | null>(null);

  React.useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    []
  );

  const onClick = async () => {
    try {
      // Fenced block (same format as core `mermaid_markdown`) so pasting
      // into a .md file renders without adding the fence by hand.
      await navigator.clipboard.writeText('```mermaid\n' + diagram + '\n```\n');
      setState('copied');
    } catch {
      setState('error');
    }
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setState('idle'), 2000);
  };

  return (
    <button
      type="button"
      onClick={onClick}
      data-testid="copy-mermaid-button"
      className="text-xs px-2 py-1 rounded border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
    >
      {state === 'copied' ? 'Copied!' : state === 'error' ? 'Copy failed' : 'Copy mermaid'}
    </button>
  );
};

export const PipelineDiagramView: React.FC<Props> = ({ jobs }) => {
  const withDiagrams = jobs.filter((job) => getDiagram(job) !== null);

  if (withDiagrams.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 text-center text-gray-500 dark:text-gray-400">
        None of the selected runs carry a pipeline diagram yet.
        Diagrams are recorded automatically for runs completed after this feature shipped.
      </div>
    );
  }

  return (
    <div
      className={`grid gap-6 ${withDiagrams.length > 1 ? 'grid-cols-1 xl:grid-cols-2' : 'grid-cols-1'}`}
      data-testid="pipeline-diagram-view"
    >
      {jobs.map((job) => {
        const diagram = getDiagram(job);
        return (
          <div
            key={job.job_id}
            className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
          >
            <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
              <span className="text-sm font-medium text-gray-800 dark:text-gray-100">
                Run {shortRunId(job)}
              </span>
              <div className="flex items-center gap-3">
                {diagram && <CopyMermaidButton diagram={diagram} />}
                {job.model_type && (
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {job.model_type}
                  </span>
                )}
              </div>
            </div>
            {diagram ? (
              <MermaidDiagram chart={diagram} className="p-4" />
            ) : (
              <div className="p-6 text-sm text-gray-400 dark:text-gray-500">
                No diagram recorded for this run (legacy job).
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
