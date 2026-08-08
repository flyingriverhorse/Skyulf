import React, { useState } from 'react';
import { describeOperationalRef } from '../../core/utils/operationalContext';
import { NodeInspectorModal, NodeInspectorTarget } from './NodeInspectorModal';

export interface NodeInspectorLinkProps {
  nodeId: string;
  /** Preferred target — a job's stored graph is queried directly. */
  jobId?: string | null;
  /** Fallback target when no `jobId` is known (e.g. a pipeline-run log entry). */
  pipelineId?: string | null;
  label?: React.ReactNode;
  origin?: string;
  filters?: Record<string, string>;
  className?: string;
}

/**
 * Opens a read-only `NodeInspectorModal` in place, instead of navigating to
 * the ML Canvas.
 *
 * Canvas navigation is a dead end whenever the pipeline currently open isn't
 * the one that produced the run, or the run was a synthetic preview/branch
 * that was never saved as a pipeline. This renders as a link for visual
 * consistency with `RecordLink`, but opens the inspector modal in place on
 * whichever operational page the user is already on.
 */
export const NodeInspectorLink: React.FC<NodeInspectorLinkProps> = ({
  nodeId,
  jobId,
  pipelineId,
  label,
  origin,
  filters,
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const description = describeOperationalRef({
    kind: 'node',
    nodeId,
    ...(pipelineId ? { pipelineId } : {}),
  });

  const target: NodeInspectorTarget | null = jobId
    ? { kind: 'job', jobId }
    : pipelineId
      ? { kind: 'pipelineRun', pipelineId }
      : null;

  if (!target) {
    return <span className="text-xs text-slate-400 italic">{description}</span>;
  }

  return (
    <>
      <button
        type="button"
        onClick={() => { setIsOpen(true); }}
        aria-label={description}
        title={description}
        className={`text-blue-600 hover:underline dark:text-blue-400 ${className}`}
      >
        {label ?? description}
      </button>
      <NodeInspectorModal
        isOpen={isOpen}
        onClose={() => { setIsOpen(false); }}
        target={target}
        nodeId={nodeId}
        {...(origin !== undefined ? { origin } : {})}
        {...(filters !== undefined ? { filters } : {})}
      />
    </>
  );
};
