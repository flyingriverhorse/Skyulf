import React from 'react';
import { ChevronDown, ChevronRight, GitBranch, Trophy } from 'lucide-react';
import type { JobInfo } from '../../../../core/api/jobs';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import {
  formatMetricName,
  getMetricDescription,
  getHyperparamDescription,
  getTrainingConfigDescription,
} from '../../../../core/utils/format';
import { getJobScoringMetric, shortRunId } from '../utils/jobMeta';

type GraphNode = {
  node_id: string;
  step_type?: string;
  params?: Record<string, unknown>;
  inputs?: string[];
};

interface Props {
  selectedJobs: JobInfo[];
  metricKeys: string[];
  isPipelineExpanded: boolean;
  setIsPipelineExpanded: (v: boolean) => void;
  isMetricsExpanded: boolean;
  setIsMetricsExpanded: (v: boolean) => void;
  isParamsExpanded: boolean;
  setIsParamsExpanded: (v: boolean) => void;
  isTuningExpanded: boolean;
  setIsTuningExpanded: (v: boolean) => void;
}

export const ComparisonTableView: React.FC<Props> = ({
  selectedJobs,
  metricKeys,
  isPipelineExpanded,
  setIsPipelineExpanded,
  isMetricsExpanded,
  setIsMetricsExpanded,
  isParamsExpanded,
  setIsParamsExpanded,
  isTuningExpanded,
  setIsTuningExpanded,
}) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">Detailed Comparison</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-900/50 dark:text-gray-400">
            <tr>
              <th className="px-4 py-2">Parameter / Metric</th>
              {selectedJobs.map(job => (
                <th key={job.job_id} className="px-4 py-2 font-mono break-all min-w-[100px]">
                  {shortRunId(job)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {/* Model Type */}
            <tr className="bg-white dark:bg-gray-800">
              <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100">Model Type</td>
              {selectedJobs.map(job => (
                <td key={job.job_id} className="px-4 py-2 text-gray-500 dark:text-gray-400">
                  <div className="flex items-center gap-1.5">
                    {job.model_type}
                    {job.branch_index != null && (
                      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 text-[10px] font-semibold">
                        <GitBranch className="w-2.5 h-2.5" /> Path {String.fromCharCode(65 + (job.branch_index ?? 0))}
                      </span>
                    )}
                    {job.promoted_at && (
                      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-[10px] font-semibold">
                        <Trophy className="w-2.5 h-2.5" /> Winner
                      </span>
                    )}
                  </div>
                </td>
              ))}
            </tr>
            {/* Pipeline Steps — preprocessing/splits/etc that fed each terminal */}
            <tr
              className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
              onClick={() => { setIsPipelineExpanded(!isPipelineExpanded); }}
            >
              <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={selectedJobs.length + 1}>
                {isPipelineExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                Pipeline Steps
              </td>
            </tr>
            {isPipelineExpanded && (() => {
              // Walk each job's graph backwards from the training
              // terminal to collect preprocessing / split / encoding
              // ancestors. Each row in the table is one position in
              // the chain (Step 1, Step 2, …) so users can compare
              // what came before each model side-by-side.
              const friendlyStep = (st: string): string => {
                if (!st) return '';
                if (st.includes('_')) {
                  return st.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                }
                return st.replace(/(?<!^)(?=[A-Z])/g, ' ').trim();
              };
              const collectChain = (job: typeof selectedJobs[number]): GraphNode[] => {
                const graphNodes = (job.graph?.nodes as GraphNode[] | undefined) || [];
                if (graphNodes.length === 0 || !job.node_id) return [];
                const map = new Map(graphNodes.map(n => [n.node_id, n]));
                // BFS backwards from the terminal, then return in
                // root → terminal order (excluding the terminal).
                const seen = new Set<string>();
                const order: GraphNode[] = [];
                const walk = (id: string): void => {
                  if (seen.has(id)) return;
                  seen.add(id);
                  const n = map.get(id);
                  if (!n) return;
                  for (const parent of n.inputs || []) walk(parent);
                  order.push(n);
                };
                walk(job.node_id);
                return order.filter(n => n.node_id !== job.node_id);
              };
              const summarizeStep = (n: GraphNode): string => {
                const display = (n.params?._display_name as string | undefined) || friendlyStep(String(n.step_type || ''));
                const interestingKeys = ['method', 'strategy', 'columns', 'target_column', 'test_size', 'val_size', 'random_state', 'n_neighbors'];
                const detail: string[] = [];
                for (const k of interestingKeys) {
                  const v = n.params?.[k];
                  if (v === undefined || v === null || v === '') continue;
                  if (Array.isArray(v) && v.length === 0) continue;
                  let rendered: string;
                  if (Array.isArray(v)) {
                    // Show real column names; truncate long lists so
                    // the cell stays readable.
                    const items = v.map(x => String(x));
                    rendered = items.length > 4
                      ? `[${items.slice(0, 4).join(', ')}, +${items.length - 4} more]`
                      : `[${items.join(', ')}]`;
                  } else if (typeof v === 'object') {
                    rendered = JSON.stringify(v);
                  } else {
                    rendered = String(v);
                  }
                  detail.push(`${k}=${rendered}`);
                }
                // Special-case Feature Generation: its config lives
                // under `operations: MathOperation[]`, not the
                // generic keys above. Render the per-op summary
                // (e.g. `multiply(a, b)`, `month(date)`) so the
                // comparison row carries real signal instead of
                // just "Feature Generation".
                const ops = n.params?.['operations'];
                if (Array.isArray(ops) && ops.length > 0) {
                  const formatted = ops.slice(0, 3).map((raw) => {
                    const op = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
                    const method = String(op['method'] ?? op['operation_type'] ?? 'op');
                    const inputs = Array.isArray(op['input_columns']) ? op['input_columns'] as unknown[] : [];
                    const secondary = Array.isArray(op['secondary_columns']) ? op['secondary_columns'] as unknown[] : [];
                    const operands = [...inputs, ...secondary].map(String);
                    const args = operands.length > 2
                      ? `${operands.slice(0, 2).join(', ')}, +${operands.length - 2}`
                      : operands.join(', ');
                    return args ? `${method}(${args})` : method;
                  });
                  const tail = ops.length > 3 ? `, +${ops.length - 3} more` : '';
                  detail.push(`ops=[${formatted.join(', ')}${tail}]`);
                }
                return detail.length > 0 ? `${display} (${detail.join(', ')})` : display;
              };
              const chainsByJob = new Map(selectedJobs.map(j => [j.job_id, collectChain(j)]));
              const allChains = Array.from(chainsByJob.values());
              if (allChains.every(c => c.length === 0)) {
                return (
                  <tr className="bg-white dark:bg-gray-800">
                    <td className="px-4 py-1.5 text-gray-400 italic pl-8" colSpan={selectedJobs.length + 1}>
                      No upstream pipeline steps captured for these runs.
                    </td>
                  </tr>
                );
              }
              // L6 — Align rows by node_id rather than by raw
              // chain index. When the trunk is shared (the
              // common case after the per-branch graph snapshot
              // fix), shared nodes occupy a single row across
              // all columns. Branch-only steps land on their
              // own rows with em-dashes in the other columns.
              // Algorithm: walk every chain in lockstep,
              // emitting the next un-emitted node id in chain
              // order. A node id appears in the merged order
              // the first time any chain references it.
              const mergedOrder: string[] = [];
              const emitted = new Set<string>();
              const cursors = allChains.map(() => 0);
              // Bound the loop to the sum of chain lengths to
              // guarantee termination on pathological inputs.
              const safety = allChains.reduce((s, c) => s + c.length, 0) + 1;
              for (let guard = 0; guard < safety; guard++) {
                let advanced = false;
                for (let ci = 0; ci < allChains.length; ci++) {
                  const chain = allChains[ci];
                  if (!chain) continue;
                  // Skip past nodes we've already emitted.
                  while (cursors[ci]! < chain.length && emitted.has(chain[cursors[ci]!]!.node_id)) {
                    cursors[ci] = cursors[ci]! + 1;
                  }
                  if (cursors[ci]! < chain.length) {
                    const candidate = chain[cursors[ci]!]!;
                    if (!emitted.has(candidate.node_id)) {
                      mergedOrder.push(candidate.node_id);
                      emitted.add(candidate.node_id);
                      advanced = true;
                      break; // restart sweep so other chains catch up
                    }
                  }
                }
                if (!advanced) break;
              }
              return mergedOrder.map((nid, idx) => {
                // Per-row cell strings, plus a sameness flag
                // for shared trunk highlighting.
                const cells = selectedJobs.map(job => {
                  const chain = chainsByJob.get(job.job_id) || [];
                  const step = chain.find(n => n.node_id === nid);
                  return step ? summarizeStep(step) : null;
                });
                const presentCells = cells.filter((c): c is string => c !== null);
                const allPresent = presentCells.length === cells.length;
                const allSame = allPresent && presentCells.every(c => c === presentCells[0]);
                // Shared trunk row → muted background; divergent
                // row (some columns dash, others differ) →
                // amber accent so the diff jumps out.
                const rowTone = allSame
                  ? 'bg-white dark:bg-gray-800'
                  : 'bg-amber-50/40 dark:bg-amber-900/10';
                return (
                  <tr key={`pipeline-step-${nid}`} className={`${rowTone} hover:bg-gray-50 dark:hover:bg-gray-700/50`}>
                    <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">Step {idx + 1}</td>
                    {cells.map((text, ci) => (
                      <td
                        key={selectedJobs[ci]?.job_id ?? ci}
                        className={
                          text === null
                            ? 'px-4 py-1.5 text-gray-300 dark:text-gray-600'
                            : allSame
                              ? 'px-4 py-1.5 text-gray-500 dark:text-gray-400'
                              : 'px-4 py-1.5 text-gray-900 dark:text-gray-100 font-medium'
                        }
                      >
                        {text ?? <span className="text-gray-400">—</span>}
                      </td>
                    ))}
                  </tr>
                );
              });
            })()}
            {/* Metrics Section in Table */}
            <tr
              className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
              onClick={() => { setIsMetricsExpanded(!isMetricsExpanded); }}
            >
              <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={selectedJobs.length + 1}>
                {isMetricsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                Key Metrics
              </td>
            </tr>
            {isMetricsExpanded && metricKeys.map(metricKey => (
              <tr key={metricKey} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">
                  <div className="flex items-center gap-1">
                    {metricKey === 'best_score'
                      ? `Best Score (${formatMetricName(getJobScoringMetric(selectedJobs[0] ?? {} as JobInfo)) || 'CV'})`
                      : metricKey.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    {getMetricDescription(metricKey) && <InfoTooltip size="sm" text={getMetricDescription(metricKey)!} />}
                  </div>
                </td>
                {selectedJobs.map(job => {
                  const m = (job.metrics || job.result?.metrics || {}) as Record<string, unknown>;
                  const val = m[metricKey];
                  return (
                    <td key={job.job_id} className="px-4 py-1.5 font-mono text-gray-600 dark:text-gray-300">
                      {typeof val === 'number' ? (metricKey.endsWith('_std') ? val.toFixed(6) : val.toFixed(4)) : '-'}
                    </td>
                  );
                })}
              </tr>
            ))}
            {/* Hyperparameters (actual model params only) */}
            <tr
              className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
              onClick={() => { setIsParamsExpanded(!isParamsExpanded); }}
            >
              <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={selectedJobs.length + 1}>
                {isParamsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                Hyperparameters
              </td>
            </tr>
            {/* Extract actual model hyperparameters: best_params for advanced, nested hyperparameters for basic */}
            {isParamsExpanded && (() => {
              const getModelParams = (job: { job_type?: string; hyperparameters?: unknown }): Record<string, unknown> => {
                const hp = job.hyperparameters as Record<string, unknown> | undefined;
                if (!hp) return {};
                if (job.job_type === 'advanced_tuning') {
                  // For advanced tuning, hyperparameters IS the best_params (or search_space) directly
                  return hp;
                }
                // Basic training: extract the nested 'hyperparameters' dict (actual model params)
                const nested = hp.hyperparameters;
                if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
                  return nested as Record<string, unknown>;
                }
                return {};
              };
              const allKeys = Array.from(new Set(selectedJobs.flatMap(job => Object.keys(getModelParams(job)))));
              if (allKeys.length === 0) {
                return (
                  <tr className="bg-white dark:bg-gray-800">
                    <td className="px-4 py-1.5 text-gray-400 dark:text-gray-500 pl-8 italic" colSpan={selectedJobs.length + 1}>
                      Default parameters (none customized)
                    </td>
                  </tr>
                );
              }
              return allKeys.map(paramKey => (
                <tr key={paramKey} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                  <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">
                    <div className="flex items-center gap-1">
                      {paramKey}
                      {getHyperparamDescription(paramKey) && <InfoTooltip size="sm" text={getHyperparamDescription(paramKey)!} />}
                    </div>
                  </td>
                  {selectedJobs.map(job => {
                    const params = getModelParams(job);
                    const val = params[paramKey];
                    return (
                      <td key={job.job_id} className="px-4 py-1.5 font-mono text-gray-600 dark:text-gray-300">
                        {val === undefined ? '-' : typeof val === 'object' ? JSON.stringify(val) : String(val)}
                      </td>
                    );
                  })}
                </tr>
              ));
            })()}

            {/* Training Configuration */}
            <tr
              className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
              onClick={() => { setIsTuningExpanded(!isTuningExpanded); }}
            >
              <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={selectedJobs.length + 1}>
                {isTuningExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                Training Configuration
              </td>
            </tr>
            {isTuningExpanded && (
              <>
                {['Target Column', 'CV Enabled', 'CV Method', 'CV Folds', 'CV Shuffle', 'CV Random State', ...(selectedJobs.some(j => j.job_type === 'advanced_tuning') ? ['Strategy', 'Strategy Params', 'Metric', 'Trials'] : [])].map(field => (
                  <tr key={field} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">
                      <div className="flex items-center gap-1">
                        {field}
                        {getTrainingConfigDescription(field) && <InfoTooltip size="sm" text={getTrainingConfigDescription(field)!} />}
                      </div>
                    </td>
                    {selectedJobs.map(job => {
                      // Resolve config: for advanced tuning use job.config or graph node params,
                      // for basic training use job.hyperparameters (which contains full node params)
                      let cfg: Record<string, unknown> | null = null;
                      if (job.job_type === 'advanced_tuning') {
                        const nodeParams = (job.config as Record<string, unknown>) ||
                          (job.graph?.nodes as Array<{ node_id: string; params?: Record<string, unknown> }> | undefined)?.find((n) => n.node_id === job.node_id)?.params;
                        cfg = nodeParams || null;
                      } else {
                        cfg = (job.hyperparameters as Record<string, unknown>) ||
                          (job.graph?.nodes as Array<{ node_id: string; params?: Record<string, unknown> }> | undefined)?.find((n) => n.node_id === job.node_id)?.params || null;
                      }

                      if (!cfg) {
                        return <td key={job.job_id} className="px-4 py-1.5 text-gray-400">-</td>;
                      }

                      // For advanced tuning, CV params are inside tuning_config
                      const tuningConfig = cfg.tuning_config as Record<string, unknown> | undefined;
                      const cvSource = (job.job_type === 'advanced_tuning' && tuningConfig ? tuningConfig : cfg) as Record<string, unknown>;
                      // Local helper: coerce unknown-typed config field to a renderable scalar.
                      const str = (v: unknown, fallback: string | number = '-'): string | number =>
                        v === undefined || v === null || v === '' ? fallback : (typeof v === 'number' ? v : String(v));

                      let value: string | number = '-';
                      if (field === 'Target Column') value = str(cfg.target_column ?? job.target_column);
                      if (field === 'CV Enabled') value = cvSource.cv_enabled ? 'Yes' : 'No';
                      if (field === 'CV Method') value = cvSource.cv_enabled ? str(cvSource.cv_type, 'Unknown') : '-';
                      if (field === 'CV Folds') value = cvSource.cv_enabled ? str(cvSource.cv_folds) : '-';
                      if (field === 'CV Shuffle') value = cvSource.cv_enabled ? (cvSource.cv_shuffle ? 'Yes' : 'No') : '-';
                      if (field === 'CV Random State') value = cvSource.cv_enabled ? str(cvSource.cv_random_state) : '-';
                      if (field === 'Strategy') value = str(tuningConfig?.strategy ?? tuningConfig?.search_strategy);
                      if (field === 'Strategy Params') {
                        const sp = tuningConfig?.strategy_params as Record<string, unknown> | undefined;
                        const strategy = String(tuningConfig?.strategy ?? tuningConfig?.search_strategy ?? '');
                        if (sp && Object.keys(sp).length > 0) {
                          value = JSON.stringify(sp);
                        } else if (strategy === 'optuna') {
                          value = 'sampler: tpe · pruner: median (defaults)';
                        } else if (strategy === 'halving_grid' || strategy === 'halving_random') {
                          value = 'factor: 3 · min: exhaust (defaults)';
                        } else {
                          value = '-';
                        }
                      }
                      if (field === 'Metric') value = str(tuningConfig?.metric);
                      if (field === 'Trials') value = str(tuningConfig?.n_trials);

                      return (
                        <td key={job.job_id} className="px-4 py-1.5 font-mono text-gray-600 dark:text-gray-300 capitalize">
                          {value}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
