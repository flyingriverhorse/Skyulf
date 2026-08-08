import React, { useCallback, useEffect, useState } from 'react';
import { ArrowDown, ArrowUp, Clock } from 'lucide-react';
import { ModalShell } from './ModalShell';
import { LoadingState } from './LoadingState';
import { EmptyState } from './EmptyState';
import { ErrorState } from './ErrorState';
import { RecordLink } from './RecordLink';
import { monitoringApi, NodeInspectorResponse, NodeNeighbor } from '../../core/api/monitoring';

/** Identifies which job's stored graph to read the node from. */
export type NodeInspectorTarget =
  | { kind: 'job'; jobId: string }
  | { kind: 'pipelineRun'; pipelineId: string };

export interface NodeInspectorModalProps {
  isOpen: boolean;
  onClose: () => void;
  target: NodeInspectorTarget;
  nodeId: string;
  origin?: string;
  filters?: Record<string, string>;
}

function formatTimestamp(value: string | null | undefined): string | null {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

/**
 * Read-only inspector for one pipeline node, sourced from a job's stored
 * graph snapshot rather than the live ML Canvas.
 *
 * Investigation from Error Log / Slow Nodes / Jobs used to route through the
 * canvas, which is a dead end whenever the pipeline currently open isn't the
 * one that produced the run, or the run was a synthetic preview/branch that
 * was never saved as a pipeline at all. This reads `TrainingJob.graph`
 * instead — the graph exactly as it executed — so investigation works
 * regardless of what's open on the canvas or whether the source pipeline
 * still exists.
 */
export const NodeInspectorModal: React.FC<NodeInspectorModalProps> = ({
  isOpen,
  onClose,
  target,
  nodeId,
  origin,
  filters,
}) => {
  const [currentNodeId, setCurrentNodeId] = useState(nodeId);
  const [data, setData] = useState<NodeInspectorResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) setCurrentNodeId(nodeId);
  }, [isOpen, nodeId, target]);

  const fetchNode = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response =
        target.kind === 'job'
          ? await monitoringApi.getJobNode(target.jobId, currentNodeId)
          : await monitoringApi.getPipelineRunNode(target.pipelineId, currentNodeId);
      setData(response);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : 'Failed to load node detail.');
    } finally {
      setLoading(false);
    }
  }, [target, currentNodeId]);

  useEffect(() => {
    if (!isOpen) return;
    void fetchNode();
  }, [isOpen, fetchNode]);

  const handleWalkTo = useCallback((neighbor: NodeNeighbor) => {
    setCurrentNodeId(neighbor.node_id);
  }, []);

  const executedAt = data ? formatTimestamp(data.finished_at ?? data.started_at) : null;

  return (
    <ModalShell isOpen={isOpen} onClose={onClose} title="Node Inspector" size="2xl">
      <div className="p-6 space-y-5">
        {loading && <LoadingState message="Loading node detail..." />}
        {!loading && error && <ErrorState error={error} onRetry={fetchNode} />}
        {!loading && !error && data && (
          <>
            {executedAt && (
              <p className="text-xs text-slate-500 dark:text-slate-400 italic">
                This is the graph as executed on {executedAt} — a historical snapshot, not live
                pipeline state.
              </p>
            )}

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div>
                <div className="text-slate-500 dark:text-slate-400">Job</div>
                <RecordLink recordRef={{ kind: 'job', jobId: data.job_id }} {...(origin !== undefined ? { origin } : {})} {...(filters !== undefined ? { filters } : {})} />
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">Dataset</div>
                <RecordLink
                  recordRef={{ kind: 'dataset', datasetId: data.dataset_source_id }}
                  label={data.dataset_name ?? data.dataset_source_id}
                  {...(origin !== undefined ? { origin } : {})}
                  {...(filters !== undefined ? { filters } : {})}
                />
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">Status</div>
                <div className="font-medium text-slate-800 dark:text-slate-200 capitalize">{data.status}</div>
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">Branch</div>
                <div className="font-medium text-slate-800 dark:text-slate-200">
                  {data.branch_index ?? '—'}
                </div>
              </div>
            </div>

            {!data.node ? (
              <EmptyState
                title="Node not found in this job's executed graph"
                description={`Node ${data.node_id} isn't present in the graph recorded for job ${data.job_id}. It may have been removed or renamed since this run.`}
              />
            ) : (
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                    {data.node.label}
                  </h3>
                  <p className="text-xs font-mono text-slate-500 dark:text-slate-400">
                    {data.node.node_id} · {data.node.step_type}
                  </p>
                  {typeof data.node.execution_seconds === 'number' && (
                    <p className="mt-1 text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1">
                      <Clock size={12} aria-hidden="true" />
                      {data.node.execution_seconds.toFixed(2)}s
                      {data.node.execution_status ? ` · ${data.node.execution_status}` : ''}
                    </p>
                  )}
                </div>

                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
                    Parameters
                  </h4>
                  {Object.keys(data.node.params).length === 0 ? (
                    <p className="text-xs text-slate-400 italic">No parameters recorded.</p>
                  ) : (
                    <pre className="text-xs font-mono bg-slate-50 dark:bg-slate-800 rounded-lg p-3 overflow-auto max-h-40 whitespace-pre-wrap">
                      {JSON.stringify(data.node.params, null, 2)}
                    </pre>
                  )}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1 flex items-center gap-1">
                      <ArrowUp size={12} aria-hidden="true" /> Upstream
                    </h4>
                    {data.node.upstream.length === 0 ? (
                      <p className="text-xs text-slate-400 italic">None — this is a source node.</p>
                    ) : (
                      <ul className="space-y-1">
                        {data.node.upstream.map((neighbor) => (
                          <li key={neighbor.node_id}>
                            <button
                              type="button"
                              onClick={() => { handleWalkTo(neighbor); }}
                              className="text-xs text-blue-600 hover:underline dark:text-blue-400"
                            >
                              {neighbor.label} ({neighbor.node_id})
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1 flex items-center gap-1">
                      <ArrowDown size={12} aria-hidden="true" /> Downstream
                    </h4>
                    {data.node.downstream.length === 0 ? (
                      <p className="text-xs text-slate-400 italic">None — this is a terminal node.</p>
                    ) : (
                      <ul className="space-y-1">
                        {data.node.downstream.map((neighbor) => (
                          <li key={neighbor.node_id}>
                            <button
                              type="button"
                              onClick={() => { handleWalkTo(neighbor); }}
                              className="text-xs text-blue-600 hover:underline dark:text-blue-400"
                            >
                              {neighbor.label} ({neighbor.node_id})
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              </div>
            )}

            {data.recent_logs.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
                  Recent log entries
                </h4>
                <ul className="space-y-1">
                  {data.recent_logs.map((log, index) => (
                    <li key={index} className="text-xs text-slate-600 dark:text-slate-300">
                      <span className="font-semibold uppercase">{log.level}</span>: {log.message}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="pt-2 border-t border-slate-100 dark:border-slate-700">
              {data.can_open_in_canvas ? (
                <RecordLink
                  recordRef={{ kind: 'node', nodeId: currentNodeId, pipelineId: data.pipeline_id }}
                  label="Open in Canvas"
                  {...(origin !== undefined ? { origin } : {})}
                  {...(filters !== undefined ? { filters } : {})}
                />
              ) : (
                <p className="text-xs text-slate-400 italic">
                  This run isn&apos;t a saved pipeline, so it can&apos;t be opened on the canvas.
                </p>
              )}
            </div>
          </>
        )}
      </div>
    </ModalShell>
  );
};
