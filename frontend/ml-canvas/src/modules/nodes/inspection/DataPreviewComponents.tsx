import React, { useState, useEffect, useMemo } from 'react';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { Activity, CheckCircle, AlertCircle, Play, GitBranch } from 'lucide-react';
import { jobsApi, JobInfo } from '../../../core/api/jobs';
import { convertGraphToPipelineConfig } from '../../../core/utils/pipelineConverter';
import { generateBranchColors } from '../../../core/hooks/useBranchColors';

export interface DataPreviewConfig {
  /** Legacy single-job id (kept for backward compatibility with old saved graphs). */
  lastRunJobId?: string | undefined;
  /** All job ids for the most recent preview run (one per parallel input branch). */
  lastRunJobIds?: string[] | undefined;
}

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (value && typeof value === 'object') return value as Record<string, unknown>;
  return null;
};

// Helper to render a mini table
const renderTable = (summary: unknown) => {
  const s = asRecord(summary);
  const sample = s?.sample;
  if (!s || !Array.isArray(sample) || sample.length === 0) {
    return <div className="text-xs text-muted-foreground italic">No data available</div>;
  }

  // Show every column the backend sent (previously capped at 5, which hid
  // newly-added columns from things like MissingIndicator). The container
  // is overflow-x-auto so wide tables scroll horizontally instead of
  // truncating silently.
  const firstRow = asRecord(sample[0]) ?? {};
  const cols = Object.keys(firstRow);

  const name = typeof s.name === 'string' ? s.name : 'Dataset';
  const shape = Array.isArray(s.shape) ? s.shape : null;
  const rows = shape && shape.length > 0 ? shape[0] : undefined;
  const colsCount = shape && shape.length > 1 ? shape[1] : undefined;
  // Cap visible rows so the panel stays compact; full row count is shown
  // in the header so users can tell the table is a sample.
  const visibleRows = sample.slice(0, 20);

  return (
    <div className="overflow-x-auto rounded border border-border max-h-72">
      <div className="text-xs font-semibold mb-1 flex justify-between p-2 bg-muted/50 border-b border-border sticky top-0 z-10">
          <span>{name}</span>
          <span className="text-muted-foreground">
            {rows ?? '?'} rows x {colsCount ?? cols.length} cols
            {visibleRows.length < sample.length && (
              <span className="ml-1 opacity-70">(showing {visibleRows.length})</span>
            )}
          </span>
      </div>
      <table className="w-full text-[10px] border-collapse">
        <thead>
          <tr className="bg-muted/30">
            {cols.map(c => <th key={c} className="p-2 border-b border-r border-border text-left font-medium text-muted-foreground last:border-r-0 whitespace-nowrap">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {visibleRows.map((row: unknown, i: number) => (
            <tr key={i} className="border-b border-border last:border-0 hover:bg-muted/20">
              {cols.map(c => (
                <td key={c} className="p-2 border-r border-border truncate max-w-[120px] last:border-r-0">
                  {String((asRecord(row)?.[c] ?? ''))}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export const DataPreviewSettings: React.FC<{ config: DataPreviewConfig; onChange: (c: DataPreviewConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId
}) => {
  const { nodes, edges } = useGraphStore();
  const [isRunning, setIsRunning] = useState(false);
  const [jobs, setJobs] = useState<Record<string, JobInfo>>({});
  const [activeBranch, setActiveBranch] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string | null>(null);

  // Resolve active job ids (new field with legacy single-id fallback).
  const jobIds = useMemo(() => {
    if (config.lastRunJobIds && config.lastRunJobIds.length > 0) return config.lastRunJobIds;
    if (config.lastRunJobId) return [config.lastRunJobId];
    return [];
  }, [config.lastRunJobIds, config.lastRunJobId]);

  // Per-branch labels derived from incoming edges in the same order as
  // convertGraphToPipelineConfig (edges.filter(e => e.target === id)),
  // so labels line up with backend job order.
  const branchLabels = useMemo(() => {
    if (!nodeId || jobIds.length <= 1) return [] as string[];
    const incoming = edges.filter(e => e.target === nodeId);
    return incoming.map((edge, idx) => {
      const letter = String.fromCharCode(65 + idx);
      const sourceNode = nodes.find(n => n.id === edge.source);
      const data = (sourceNode?.data ?? {}) as Record<string, unknown>;
      const suffix =
        (data.label as string) ||
        (data.title as string) ||
        (typeof data.definitionType === 'string'
          ? (data.definitionType as string).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
          : '');
      return suffix ? `Path ${letter} · ${suffix}` : `Path ${letter}`;
    });
  }, [nodes, edges, nodeId, jobIds.length]);

  // Poll all jobs until each reaches a terminal state.
  useEffect(() => {
    if (jobIds.length === 0) return;
    let cancelled = false;
    const fetchAll = async () => {
      const updated: Record<string, JobInfo> = {};
      await Promise.all(
        jobIds.map(async (id) => {
          try {
            updated[id] = await jobsApi.getJob(id);
          } catch (e) {
            console.error('Failed to fetch job', id, e);
          }
        }),
      );
      if (!cancelled) setJobs(updated);
    };
    void fetchAll();
    const interval = setInterval(() => { void fetchAll(); }, 2000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [jobIds]);

  const handleRunPreview = async () => {
    if (!nodeId) return;
    setIsRunning(true);
    try {
      const pipelineConfig = convertGraphToPipelineConfig(nodes, edges);

      const response = await jobsApi.runPipeline({
        ...pipelineConfig,
        target_node_id: nodeId,
        job_type: 'preview'
      });

      const ids = response.job_ids?.length ? response.job_ids : [response.job_id];
      setJobs({});
      setActiveBranch(null);
      setActiveTab(null);
      onChange({ ...config, lastRunJobId: ids[0], lastRunJobIds: ids });
    } catch (error) {
      console.error('Failed to run preview:', error);
      alert('Failed to start preview job.');
    } finally {
      setIsRunning(false);
    }
  };

  // Per-branch entries keyed by label. Each parallel job runs the same
  // data_preview node against a different upstream path.
  const branchResults = useMemo(() => {
    return jobIds.map((id, idx) => ({
      label: branchLabels[idx] || (jobIds.length > 1 ? `Path ${String.fromCharCode(65 + idx)}` : 'Result'),
      jobId: id,
      job: jobs[id] as JobInfo | undefined,
    }));
  }, [jobIds, branchLabels, jobs]);

  const branchColors = useMemo(
    () => generateBranchColors(branchResults.length),
    [branchResults.length],
  );

  // Default to the first branch once jobs land; reset when wiring changes.
  useEffect(() => {
    if (branchResults.length > 0) {
      if (!activeBranch || !branchResults.some(b => b.label === activeBranch)) {
        setActiveBranch(branchResults[0]!.label);
      }
    } else if (activeBranch !== null) {
      setActiveBranch(null);
    }
  }, [branchResults, activeBranch]);

  const activeBranchEntry = branchResults.find(b => b.label === activeBranch) || branchResults[0];
  const activeJob = activeBranchEntry?.job;
  const rawResult = activeJob?.result?.metrics ?? activeJob?.result;
  const resultRec = asRecord(rawResult);
  const dataSummary = resultRec ? asRecord(resultRec.data_summary) : null;
  const operationMode = resultRec?.operation_mode;
  const operationModeText = typeof operationMode === 'string' ? operationMode : undefined;

  const splitTabs = useMemo(() => {
    if (!dataSummary) return [] as string[];
    return ['train', 'test', 'validation', 'full'].filter(t => dataSummary[t]);
  }, [dataSummary]);

  // Reset split tab when active branch changes or available tabs shift.
  useEffect(() => {
    if (splitTabs.length === 0) {
      if (activeTab !== null) setActiveTab(null);
      return;
    }
    if (!activeTab || !splitTabs.includes(activeTab)) {
      setActiveTab(splitTabs[0] ?? null);
    }
  }, [splitTabs, activeTab]);

  // Aggregate status: failed > running > completed.
  const aggregateStatus: 'completed' | 'failed' | 'running' | 'idle' = useMemo(() => {
    if (branchResults.length === 0) return 'idle';
    const statuses = branchResults.map(b => b.job?.status);
    if (statuses.some(s => s === 'failed')) return 'failed';
    if (statuses.every(s => s === 'completed')) return 'completed';
    return 'running';
  }, [branchResults]);

  return (
    <div className="p-4 space-y-4 h-full overflow-y-auto">
      <div className="text-sm text-muted-foreground">
        Connect this node to any data output to inspect the data state, schema, and applied transformations.
        Connecting two or more upstream paths shows each one in its own tab.
      </div>

      <button
        onClick={() => { void handleRunPreview(); }}
        disabled={isRunning}
        className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-[#38bdf8] via-[#6366f1] to-[#a855f7] text-white px-3 py-1.5 rounded-md hover:opacity-90 disabled:opacity-50 transition-all text-xs font-medium shadow-md"
      >
        {isRunning ? <Activity size={14} className="animate-spin" /> : <Play size={14} />}
        {isRunning ? 'Running...' : 'Run Preview'}
      </button>

      {jobIds.length > 0 && aggregateStatus !== 'completed' && (
        <div className="text-xs text-muted-foreground flex items-center gap-2 p-2 bg-muted/30 rounded border border-border">
          <span>Status:</span>
          <span className={`ml-auto font-medium px-1.5 py-0.5 rounded text-[10px] ${
            aggregateStatus === 'failed' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
            'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
          }`}>
            {aggregateStatus}
          </span>
          {branchResults.length > 1 && (
            <span className="text-[10px] opacity-80">
              {branchResults.filter(b => b.job?.status === 'completed').length}/{branchResults.length} branches
            </span>
          )}
        </div>
      )}

      {/* Results Area */}
      {aggregateStatus !== 'idle' && rawResult && (
        <div className="space-y-3 border-t border-border pt-4 animate-in fade-in slide-in-from-top-2 duration-300">
            {/* Branch Tabs (multi-input previews only) */}
            {branchResults.length > 1 && (
              <div className="flex flex-wrap items-center gap-1 border-b border-border pb-1">
                <GitBranch size={12} className="text-muted-foreground mr-1" />
                {branchResults.map((b, idx) => {
                  const isActive = activeBranch === b.label;
                  const status = b.job?.status;
                  return (
                    <button
                      key={b.label}
                      onClick={() => { setActiveBranch(b.label); setActiveTab(null); }}
                      className={`flex items-center gap-1.5 px-2 py-1 text-[10px] font-medium rounded-md border transition-colors ${
                        isActive
                          ? 'bg-background text-foreground border-border'
                          : 'bg-muted/30 text-muted-foreground hover:bg-muted/50 border-transparent'
                      }`}
                      title={status ? `${b.label} — ${status}` : b.label}
                    >
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{ backgroundColor: branchColors[idx] }}
                      />
                      {b.label}
                      {status === 'failed' && <AlertCircle size={10} className="text-red-500" />}
                      {status === 'completed' && <CheckCircle size={10} className="text-green-500" />}
                      {status && status !== 'completed' && status !== 'failed' && (
                        <Activity size={10} className="animate-spin" />
                      )}
                    </button>
                  );
                })}
              </div>
            )}

            {/* Operation Mode */}
            <div className="text-[10px] bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-300 p-2 rounded border border-blue-100 dark:border-blue-900/50">
                <strong className="block mb-0.5">Operation Mode</strong>
                {operationModeText || 'Unknown'}
            </div>

            {/* Split Tabs */}
            {dataSummary && splitTabs.length > 0 && (
                <div className="space-y-2">
                    <div className="flex border-b border-border">
                        {splitTabs.map((t) => (
                          <button
                            key={t}
                            className={`text-[10px] px-3 py-1.5 border-b-2 transition-colors ${
                              activeTab === t
                                ? 'border-primary font-medium text-primary'
                                : 'border-transparent text-muted-foreground hover:text-foreground'
                            }`}
                            onClick={() => { setActiveTab(t); }}
                          >
                            {t.charAt(0).toUpperCase() + t.slice(1)}
                          </button>
                        ))}
                    </div>

                    {activeTab && dataSummary[activeTab]
                      ? renderTable(dataSummary[activeTab])
                      : (
                        <div className="text-xs text-muted-foreground p-4 text-center border border-dashed border-border rounded">
                          No {activeTab ?? 'data'} available
                        </div>
                      )}
                </div>
            )}

            {/* Per-branch failure surface */}
            {activeJob?.status === 'failed' && (
              <div className="text-[10px] bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-400 p-2 rounded border border-red-100 dark:border-red-900/50">
                <strong className="block mb-0.5">Branch failed</strong>
                <span className="font-mono whitespace-pre-wrap">{activeJob.error || 'Unknown error'}</span>
              </div>
            )}

            {/* Transformations */}
            {Array.isArray(resultRec?.applied_transformations) && resultRec.applied_transformations.length > 0 && (
                <div className="pt-2 border-t border-border">
                    <div className="text-[10px] font-semibold mb-2 text-foreground">Applied Steps</div>
                    <div className="space-y-1.5">
                  {(resultRec.applied_transformations as unknown[]).map((t: unknown, i: number) => (
                            <div key={i} className="text-[10px] flex items-center gap-2 text-muted-foreground p-1.5 bg-muted/30 rounded border border-border">
                                <span className="w-4 h-4 rounded-full bg-muted flex items-center justify-center text-[8px] font-medium text-foreground">{i+1}</span>
                                <div className="flex flex-col">
                        <span className="font-medium text-foreground">{String(asRecord(t)?.transformer_name ?? '')}</span>
                        <span className="text-[9px] opacity-80">{String(asRecord(t)?.transformer_type ?? '')}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
      )}
    </div>
  );
};

export const DataPreviewComponent: React.FC<{ data: DataPreviewConfig }> = ({ data }) => {
  const [status, setStatus] = useState<string | null>(null);

  // Aggregate status across all branch jobs (parallel previews) so the node
  // badge reflects the overall preview state instead of just the first job.
  useEffect(() => {
    const ids = data.lastRunJobIds && data.lastRunJobIds.length > 0
      ? data.lastRunJobIds
      : (data.lastRunJobId ? [data.lastRunJobId] : []);
    if (ids.length === 0) return;

    const checkStatus = async () => {
      try {
        const results = await Promise.all(ids.map(id => jobsApi.getJob(id)));
        const statuses = results.map(r => r.status);
        if (statuses.some(s => s === 'failed')) setStatus('failed');
        else if (statuses.every(s => s === 'completed')) setStatus('completed');
        else setStatus('running');
      } catch {
        setStatus('error');
      }
    };

    void checkStatus();
    const interval = setInterval(() => { void checkStatus(); }, 5000);

    return () => { clearInterval(interval); };
  }, [data.lastRunJobId, data.lastRunJobIds]);

  const hasJob = (data.lastRunJobIds && data.lastRunJobIds.length > 0) || !!data.lastRunJobId;

  return (
    <div className="text-xs flex items-center gap-2">
       {!hasJob ? (
         <span className="text-muted-foreground italic">Not run yet</span>
       ) : status === 'completed' ? (
         <span className="text-green-600 dark:text-green-400 font-medium flex items-center gap-1"><CheckCircle size={12}/> Ready</span>
       ) : status === 'failed' ? (
         <span className="text-red-600 dark:text-red-400 font-medium flex items-center gap-1"><AlertCircle size={12}/> Failed</span>
       ) : (
         <span className="text-blue-600 dark:text-blue-400 font-medium flex items-center gap-1"><Activity size={12} className="animate-spin"/> Running</span>
       )}
    </div>
  );
};
