import React, { useState, useRef, useMemo, useEffect, useCallback } from 'react';
import {
  X, ArrowLeft, Terminal, LayoutDashboard, FileText,
  AlertCircle, CheckCircle, Square, Database, Copy, WrapText,
  ChevronsDown, ChevronsUp,
} from 'lucide-react';
import { JobInfo } from '../../../core/api/jobs';
import { useJobStore } from '../../../core/store/useJobStore';
import { useJobPolling, isTerminalStatus } from '../../../core/hooks/useJobPolling';
import { formatMetricName, getMetricDescription } from '../../../core/utils/format';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { useConfirm } from '../../shared';
import { toast } from '../../../core/toast';

/** Parse a log line into its level and message parts. */
type LogLevel = 'error' | 'warning' | 'info' | 'debug' | 'plain';

interface ParsedLog {
  level: LogLevel;
  prefix: string;
  message: string;
  raw: string;
}

function parseLogLine(raw: string): ParsedLog {
  // Match Python logging: "INFO:logger:message", "WARNING:logger:message", etc.
  const pyLogging = /^(DEBUG|INFO|WARNING|ERROR|CRITICAL):([^:]+):(.*)/i.exec(raw);
  if (pyLogging) {
    const lvl = pyLogging[1]!.toUpperCase();
    const level: LogLevel = lvl === 'ERROR' || lvl === 'CRITICAL' ? 'error' : lvl === 'WARNING' ? 'warning' : lvl === 'DEBUG' ? 'debug' : 'info';
    return { level, prefix: `${pyLogging[1]}:${pyLogging[2]}`, message: pyLogging[3] ?? '', raw };
  }
  // Match bracket style: "[INFO]", "[ERROR]", "[WARNING]"
  const bracket = /^\[(INFO|WARNING|WARN|ERROR|DEBUG)\]\s*(.*)/i.exec(raw);
  if (bracket) {
    const lvl = bracket[1]!.toUpperCase();
    const level: LogLevel = lvl === 'ERROR' ? 'error' : lvl === 'WARNING' || lvl === 'WARN' ? 'warning' : lvl === 'DEBUG' ? 'debug' : 'info';
    return { level, prefix: `[${bracket[1]}]`, message: bracket[2] ?? '', raw };
  }
  // Match lines containing keywords
  const lower = raw.toLowerCase();
  if (/\b(error|exception|traceback|failed)\b/.test(lower)) return { level: 'error', prefix: '', message: raw, raw };
  if (/\b(warning|warn)\b/.test(lower)) return { level: 'warning', prefix: '', message: raw, raw };
  return { level: 'plain', prefix: '', message: raw, raw };
}

const LOG_LEVEL_STYLES: Record<LogLevel, { row: string; prefix: string; lineNo: string }> = {
  error:   { row: 'bg-red-950/40 hover:bg-red-950/60',   prefix: 'text-red-400 font-bold',   lineNo: 'text-red-600/60' },
  warning: { row: 'bg-yellow-950/30 hover:bg-yellow-950/50', prefix: 'text-yellow-400 font-bold', lineNo: 'text-yellow-600/60' },
  info:    { row: 'hover:bg-gray-800/50',                  prefix: 'text-blue-400',             lineNo: 'text-gray-600' },
  debug:   { row: 'hover:bg-gray-800/30',                  prefix: 'text-gray-500',             lineNo: 'text-gray-700' },
  plain:   { row: 'hover:bg-gray-800/40',                  prefix: '',                          lineNo: 'text-gray-600' },
};

// ---------- inline log token highlighting ----------

interface Segment { text: string; cls?: string }

// Rules matched left-to-right, first match wins at each position.
// Each rule may produce one or multiple Segments so key=value pairs get split colouring.
type HighlightRule = {
  re: RegExp;
  seg: (m: RegExpExecArray) => Segment[];
};

const LOG_HIGHLIGHT_RULES: HighlightRule[] = [
  // ISO timestamp (drop microseconds / tz suffix for matching purposes)
  {
    re: /\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}/,
    seg: m => [{ text: m[0], cls: 'text-slate-500' }],
  },
  // key=float  e.g.  loss=0.456  accuracy=0.9234  val_f1=0.812
  {
    re: /\b(\w+)(=)(-?\d+\.?\d*(?:[eE][+-]?\d+)?%?)/,
    seg: m => [
      { text: m[1]!, cls: 'text-sky-400' },
      { text: '=',   cls: 'text-gray-600' },
      { text: m[3]!, cls: 'text-emerald-400' },
    ],
  },
  // "Executing node: NodeName"  e.g.  Executing node: FeatureGenerationNode
  {
    re: /(Executing node:\s*)(\S+)/,
    seg: m => [
      { text: m[1]!, cls: 'text-violet-400 font-semibold' },
      { text: m[2]!, cls: 'text-violet-300 font-semibold' },
    ],
  },
  // Fold N/M or Epoch N/M progress  e.g.  Fold 2/5  Epoch 4/20  Step 100/500
  {
    re: /\b((?:fold|epoch|step|cv fold)\s*)(\d+)(\s*\/\s*)(\d+)/i,
    seg: m => [
      { text: m[1]!, cls: 'text-orange-400' },
      { text: m[2]!, cls: 'text-orange-300 font-semibold' },
      { text: m[3]!, cls: 'text-gray-500' },
      { text: m[4]!, cls: 'text-orange-300' },
    ],
  },
  // Cross-validation keywords  e.g.  cross_val_score  cross-validation  cv_results
  {
    re: /\b(?:cross[_-]val(?:idation)?(?:_\w+)?|cross_validate)\b/i,
    seg: m => [{ text: m[0], cls: 'text-orange-400 font-semibold' }],
  },
  // key=word  e.g.  status=completed  mode=train
  {
    re: /\b(\w+)(=)([A-Za-z_]\w*)/,
    seg: m => [
      { text: m[1]!, cls: 'text-sky-400' },
      { text: '=',   cls: 'text-gray-600' },
      { text: m[3]!, cls: 'text-cyan-300' },
    ],
  },
  // error / fail keywords
  {
    re: /\b(?:error|exception|traceback|failed|fatal|critical)\b/i,
    seg: m => [{ text: m[0], cls: 'text-red-400 font-semibold' }],
  },
  // success keywords
  {
    re: /\b(?:success|completed|done|finished|passed|saved)\b/i,
    seg: m => [{ text: m[0], cls: 'text-green-400 font-semibold' }],
  },
  // warning keywords
  {
    re: /\b(?:warning|warn|deprecated)\b/i,
    seg: m => [{ text: m[0], cls: 'text-yellow-400' }],
  },
  // durations with units  e.g.  3.14s  250ms  0.05s
  {
    re: /-?\d+\.?\d*(?:[eE][+-]?\d+)?\s*(?:ms|s)\b/,
    seg: m => [{ text: m[0], cls: 'text-teal-400' }],
  },
  // percentages  e.g.  98.5%
  {
    re: /-?\d+\.?\d*%/,
    seg: m => [{ text: m[0], cls: 'text-emerald-400' }],
  },
  // floats (bare)
  {
    re: /-?\d+\.\d+(?:[eE][+-]?\d+)?/,
    seg: m => [{ text: m[0], cls: 'text-emerald-400' }],
  },
  // integers (bare — low priority to avoid clobbering the above)
  {
    re: /\b-?\d+\b/,
    seg: m => [{ text: m[0], cls: 'text-emerald-300' }],
  },
  // double-quoted strings
  {
    re: /"[^"]*"/,
    seg: m => [{ text: m[0], cls: 'text-amber-300/80' }],
  },
];

function tokenizeLogMessage(msg: string): Segment[] {
  const result: Segment[] = [];
  let remaining = msg;

  while (remaining.length > 0) {
    let bestIdx = remaining.length; // default: no match
    let bestSegs: Segment[] | null = null;
    let bestLen = 0;

    for (const rule of LOG_HIGHLIGHT_RULES) {
      const m = rule.re.exec(remaining);
      if (m && m.index < bestIdx) {
        bestIdx = m.index;
        bestSegs = rule.seg(m);
        bestLen = m[0].length;
      }
    }

    if (!bestSegs) {
      // No rule matched anything — push the rest as plain text
      result.push({ text: remaining });
      break;
    }

    if (bestIdx > 0) {
      result.push({ text: remaining.slice(0, bestIdx) });
    }
    for (const s of bestSegs) result.push(s);
    remaining = remaining.slice(bestIdx + bestLen);
  }

  return result;
}

const LogMessageContent: React.FC<{ message: string }> = ({ message }) => {
  const segments = useMemo(() => tokenizeLogMessage(message), [message]);
  return (
    <>
      {segments.map((seg, i) =>
        seg.cls
          ? <span key={i} className={seg.cls}>{seg.text}</span>
          : <React.Fragment key={i}>{seg.text}</React.Fragment>
      )}
    </>
  );
};

// ---------------------------------------------------

const formatMetricValue = (key: string, value: number): string => {
  if (key.endsWith('_std')) return value.toFixed(6);
  return value.toFixed(4);
};

const getScoringMetric = (job: JobInfo): string | undefined => {
  const result = job.result as Record<string, unknown> | undefined;
  if (result?.scoring_metric) return result.scoring_metric as string;
  const config = job.config as Record<string, unknown> | undefined;
  const tuning = config?.tuning_config as Record<string, unknown> | undefined;
  return tuning?.metric as string | undefined;
};

const FeatureImportancesSection: React.FC<{ result: Record<string, unknown> }> = ({ result }) => {
  const metrics = result.metrics as Record<string, unknown> | undefined;
  const raw = (metrics?.feature_importances ?? result.feature_importances) as Record<string, number> | undefined;
  if (!raw || typeof raw !== 'object') return null;

  const sorted = Object.entries(raw).sort(([, a], [, b]) => b - a).slice(0, 5);
  if (sorted.length === 0) return null;
  const maxVal = sorted[0]![1] || 1;

  return (
    <div className="space-y-2">
      <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Feature Importances</h4>
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-3 max-h-64 overflow-y-auto">
        {sorted.map(([feature, importance]) => (
          <div key={feature} className="flex items-center gap-2 py-1">
            <span className="text-xs text-gray-600 dark:text-gray-300 w-32 truncate shrink-0" title={feature}>{feature}</span>
            <div className="flex-1 h-4 bg-gray-100 dark:bg-gray-700 rounded overflow-hidden">
              <div
                className="h-full bg-blue-500 dark:bg-blue-400 rounded"
                style={{ width: `${(importance / maxVal) * 100}%` }}
              />
            </div>
            <span className="text-xs font-mono text-gray-500 dark:text-gray-400 w-14 text-right shrink-0">{importance.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

interface JobDetailsViewProps {
  job: JobInfo;
  onBack: () => void;
  onClose: () => void;
}

export const JobDetailsView: React.FC<JobDetailsViewProps> = ({ job: initialJob, onBack, onClose }) => {
    const { cancelJob } = useJobStore();
    const confirm = useConfirm();
    const [activeTab, setActiveTab] = useState<'overview' | 'logs'>('overview');
    const [isCancelling, setIsCancelling] = useState(false);
    const [autoScroll, setAutoScroll] = useState(true);
    const [wrapLines, setWrapLines] = useState(true);
    const [copied, setCopied] = useState(false);
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Poll the single job until terminal. We feed an empty array once
    // the initial job is already terminal so the hook does no work,
    // and otherwise let it stop itself on the next terminal snapshot.
    const pollIds = useMemo(
        () => (isTerminalStatus(initialJob.status) ? [] : [initialJob.job_id]),
        [initialJob.job_id, initialJob.status],
    );
    const { jobs: polledJobs } = useJobPolling(pollIds, { intervalMs: 2000 });
    const job: JobInfo = polledJobs[initialJob.job_id] ?? initialJob;

    // Auto-scroll logs
    useEffect(() => {
        if (autoScroll && activeTab === 'logs' && logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [job.logs, activeTab, autoScroll]);

    const handleCopyLogs = useCallback(() => {
        const text = (job.logs ?? []).join('\n');
        void navigator.clipboard.writeText(text).then(() => {
            setCopied(true);
            setTimeout(() => { setCopied(false); }, 2000);
        });
    }, [job.logs]);

    const handleCancel = async () => {
        const ok = await confirm({
            title: 'Stop job?',
            message: 'Are you sure you want to stop this job?',
            confirmLabel: 'Stop',
            variant: 'danger',
        });
        if (!ok) return;
        setIsCancelling(true);
        try {
            await cancelJob(job.job_id);
        } catch (e) {
            toast.error('Failed to cancel job');
        } finally {
            setIsCancelling(false);
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-gray-800/50">
                <div className="flex items-center gap-3">
                    <button onClick={onBack} className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded text-gray-500">
                        <ArrowLeft className="w-4 h-4" />
                    </button>
                    <div>
                        <h2 className="font-semibold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                            Job Details
                            <span className="text-xs font-normal text-gray-500 font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded">
                                {job.job_id.slice(0, 8)}
                            </span>
                        </h2>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {(job.status === 'running' || job.status === 'queued') && (
                        <button 
                            onClick={() => { void handleCancel(); }}
                            disabled={isCancelling}
                            className="flex items-center gap-1 px-3 py-1.5 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/40 rounded text-xs font-medium transition-colors border border-red-200 dark:border-red-800"
                        >
                            <Square className="w-3 h-3 fill-current" />
                            {isCancelling ? 'Stopping...' : 'Stop Job'}
                        </button>
                    )}
                    <button onClick={onClose} className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400">
                        <X className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-gray-200 dark:border-gray-700 px-4">
                <button
                    className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
                    activeTab === 'overview' 
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                        : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                    }`}
                    onClick={() => { setActiveTab('overview'); }}
                >
                    <LayoutDashboard className="w-4 h-4" />
                    Overview
                </button>
                <button
                    className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
                    activeTab === 'logs' 
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                        : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                    }`}
                    onClick={() => { setActiveTab('logs'); }}
                >
                    <FileText className="w-4 h-4" />
                    Live Logs
                    {job.status === 'running' && <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />}
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
                {activeTab === 'overview' ? (
                    <div className="space-y-6">
                        {/* Status Section */}
                        <div className="grid grid-cols-3 gap-4">
                            <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Status</div>
                                <div className="font-medium capitalize flex items-center gap-2 text-gray-800 dark:text-gray-200">
                                    {job.status}
                                </div>
                            </div>
                            <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Dataset</div>
                                <div className="font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                                    <Database className="w-3 h-3 text-gray-400" />
                                    {job.dataset_name || job.dataset_id || 'Unknown'}
                                </div>
                            </div>
                            <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Duration</div>
                                <div className="font-medium text-gray-800 dark:text-gray-200 font-mono">
                                    {job.start_time && job.end_time 
                                        ? `${Math.round((new Date(job.end_time).getTime() - new Date(job.start_time).getTime()) / 1000)}s` 
                                        : '-'}
                                </div>
                            </div>
                        </div>

                        {/* Error Section */}
                        {job.error && (
                            <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-100 dark:border-red-900/30 rounded-lg">
                                <h3 className="text-sm font-medium text-red-800 dark:text-red-300 mb-2 flex items-center gap-2">
                                    <AlertCircle className="w-4 h-4" />
                                    Error Log
                                </h3>
                                <pre className="text-xs text-red-700 dark:text-red-400 whitespace-pre-wrap font-mono">
                                    {job.error}
                                </pre>
                            </div>
                        )}

                        {/* Results Section */}
                        {job.result && (
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-sm font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                                        <Terminal className="w-4 h-4" />
                                        Execution Results
                                    </h3>
                                    {job.status === 'completed' && (
                                        <span className="text-xs px-2 py-1 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded-full flex items-center gap-1 border border-green-200 dark:border-green-800 font-medium">
                                            <CheckCircle className="w-3 h-3" /> Model Ready
                                        </span>
                                    )}
                                </div>
                                
                                {job.job_type === 'basic_training' && !!(job.result as Record<string, unknown>).metrics && (
                                    <div className="space-y-4">
                                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                            {Object.entries((job.result as Record<string, unknown>).metrics as Record<string, unknown>)
                                                .filter(([, v]) => typeof v === 'number' || typeof v === 'string')
                                                .map(([k, v]) => (
                                                <div key={k} className={`p-3 border rounded-lg ${k.startsWith('cv_') ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800' : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700'}`}>
                                                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1 capitalize flex items-center gap-1">
                                                        {k.replace(/_/g, ' ')}
                                                        {getMetricDescription(k) && <InfoTooltip size="sm" text={getMetricDescription(k)!} />}
                                                    </div>
                                                    <div className={`font-mono font-medium ${k.startsWith('cv_') ? 'text-purple-600 dark:text-purple-400' : 'text-blue-600 dark:text-blue-400'}`}>
                                                        {typeof v === 'number' ? formatMetricValue(k, v) : String(v)}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        <FeatureImportancesSection result={job.result as Record<string, unknown>} />
                                    </div>
                                )}

                                {job.job_type === 'advanced_tuning' && (
                                    <div className="space-y-4">
                                        {/* Tuning Configuration */}
                                        {job.graph && (
                                            <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                                                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">Tuning Configuration</h4>
                                                <div className="grid grid-cols-2 gap-4 text-xs">
                                                    {(() => {
                                                        const node = (job.graph?.nodes as Array<{ node_id: string; params?: Record<string, unknown> }> | undefined)?.find((n) => n.node_id === job.node_id);
                                                        // Tuning config is a free-form server payload that varies by strategy.
                                                        type TuningConfig = {
                                                            strategy?: string;
                                                            search_strategy?: string;
                                                            strategy_params?: Record<string, unknown>;
                                                            metric?: string;
                                                            n_trials?: number;
                                                            cv_enabled?: boolean;
                                                            cv_type?: string;
                                                            cv_folds?: number;
                                                            cv_shuffle?: boolean;
                                                        };
                                                        const jobConfig = job.config as { tuning_config?: TuningConfig } | undefined;
                                                        const rawConfig = Object.keys(jobConfig?.tuning_config || {}).length > 0
                                                            ? jobConfig?.tuning_config
                                                            : (node?.params?.tuning_config as TuningConfig | undefined);
                                                        const config: TuningConfig | undefined = rawConfig;
                                                        if (!config) return <div className="text-gray-400 col-span-2">No configuration found</div>;
                                                        
                                                        const activeStrategy = config.strategy || config.search_strategy || '';
                                                        const sp = config.strategy_params;
                                                        const hasStrategyParams = sp && Object.keys(sp).length > 0;
                                                        const strategyParamsDisplay = hasStrategyParams
                                                            ? Object.entries(sp!).map(([k, v]) => `${k}: ${v}`).join(' · ')
                                                            : activeStrategy === 'optuna'
                                                            ? 'sampler: tpe · pruner: median (defaults)'
                                                            : activeStrategy === 'halving_grid' || activeStrategy === 'halving_random'
                                                            ? 'factor: 3 · resource: n_samples · min_resources: exhaust (defaults)'
                                                            : '-';

                                                        return (
                                                            <>
                                                                <div>
                                                                    <span className="text-gray-500">Strategy:</span>
                                                                    <span className="ml-2 font-mono text-gray-700 dark:text-gray-300 capitalize">{activeStrategy || '-'}</span>
                                                                    
                                                                </div>
                                                                <div className="col-span-2">
                                                                    <span className="text-gray-500">Strategy Params:</span>
                                                                    <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{strategyParamsDisplay}</span>
                                                                </div>
                                                                <div>
                                                                    <span className="text-gray-500">Metric:</span>
                                                                    <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.metric || '-'}</span>
                                                                </div>
                                                                <div>
                                                                    <span className="text-gray-500">Trials:</span>
                                                                    <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.n_trials || '-'}</span>
                                                                </div>
                                                                <div>
                                                                    <span className="text-gray-500">CV Enabled:</span>
                                                                    <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_enabled ? 'Yes' : 'No'}</span>
                                                                </div>
                                                                {config.cv_enabled && (
                                                                    <>
                                                                        <div>
                                                                            <span className="text-gray-500">CV Method:</span>
                                                                            <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_type || 'Unknown'}</span>
                                                                        </div>
                                                                        <div>
                                                                            <span className="text-gray-500">Folds:</span>
                                                                            <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_folds}</span>
                                                                        </div>
                                                                        <div>
                                                                            <span className="text-gray-500">Shuffle:</span>
                                                                            <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_shuffle ? 'Yes' : 'No'}</span>
                                                                        </div>
                                                                    </>
                                                                )}
                                                            </>
                                                        );
                                                    })()}
                                                </div>
                                            </div>
                                        )}

                                        {/* Best Score */}
                                        {(job.result as Record<string, unknown>).best_score !== undefined && (
                                            <div className="p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg w-fit">
                                                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                                                    Best Score{getScoringMetric(job) ? ` (${formatMetricName(getScoringMetric(job))})` : ''}
                                                </div>
                                                <div className="font-mono font-bold text-lg text-purple-600 dark:text-purple-400">
                                                    {Number((job.result as Record<string, unknown>).best_score).toFixed(4)}
                                                </div>
                                            </div>
                                        )}

                                        {/* Full Metrics (Train/Test/Val) */}
                                        {!!(job.result as Record<string, unknown>).metrics && (
                                            <div className="space-y-2">
                                                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Evaluation Metrics</h4>
                                                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                                    {Object.entries((job.result as Record<string, unknown>).metrics as Record<string, unknown>)
                                                        .filter(([k, v]) => !['best_score', 'best_params', 'trials'].includes(k) && (typeof v === 'number' || typeof v === 'string'))
                                                        .map(([k, v]) => (
                                                        <div key={k} className={`p-3 border rounded-lg ${k.startsWith('cv_') ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800' : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700'}`}>
                                                            <div className="text-xs text-gray-500 dark:text-gray-400 mb-1 capitalize flex items-center gap-1">
                                                                {k.replace(/_/g, ' ')}
                                                                {getMetricDescription(k) && <InfoTooltip size="sm" text={getMetricDescription(k)!} />}
                                                            </div>
                                                            <div className={`font-mono font-medium ${k.startsWith('cv_') ? 'text-purple-600 dark:text-purple-400' : 'text-blue-600 dark:text-blue-400'}`}>
                                                                {typeof v === 'number' ? formatMetricValue(k, v as number) : String(v)}
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                            <FeatureImportancesSection result={job.result as Record<string, unknown>} />
                                        
                                        {/* Best Params */}
                                        {!!(job.result as Record<string, unknown>).best_params && (
                                            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-xs overflow-x-auto">
                                                <div className="text-gray-500 mb-2"># Best Hyperparameters</div>
                                                <pre>{JSON.stringify((job.result as Record<string, unknown>).best_params, null, 2)}</pre>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="flex flex-col h-full min-h-[400px]">
                        {/* Log toolbar */}
                        <div className="flex items-center justify-between px-3 py-1.5 bg-gray-900 border-b border-gray-700 rounded-t-lg shrink-0">
                            <div className="flex items-center gap-1 text-xs text-gray-500">
                                <span className="font-mono">{(job.logs ?? []).length} lines</span>
                                {job.status === 'running' && (
                                    <span className="ml-2 flex items-center gap-1 text-green-400">
                                        <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse inline-block" />
                                        live
                                    </span>
                                )}
                            </div>
                            <div className="flex items-center gap-1">
                                <button
                                    onClick={() => { setWrapLines(w => !w); }}
                                    title={wrapLines ? 'No wrap' : 'Wrap lines'}
                                    className={`p-1 rounded transition-colors ${wrapLines ? 'text-blue-400 bg-blue-900/30' : 'text-gray-500 hover:text-gray-300'}`}
                                >
                                    <WrapText className="w-3.5 h-3.5" />
                                </button>
                                <button
                                    onClick={() => { setAutoScroll(a => !a); }}
                                    title={autoScroll ? 'Disable auto-scroll' : 'Enable auto-scroll'}
                                    className={`p-1 rounded transition-colors ${autoScroll ? 'text-blue-400 bg-blue-900/30' : 'text-gray-500 hover:text-gray-300'}`}
                                >
                                    {autoScroll ? <ChevronsDown className="w-3.5 h-3.5" /> : <ChevronsUp className="w-3.5 h-3.5" />}
                                </button>
                                <button
                                    onClick={handleCopyLogs}
                                    title="Copy all logs"
                                    className="p-1 rounded text-gray-500 hover:text-gray-300 transition-colors"
                                >
                                    {copied
                                        ? <CheckCircle className="w-3.5 h-3.5 text-green-400" />
                                        : <Copy className="w-3.5 h-3.5" />
                                    }
                                </button>
                            </div>
                        </div>

                        {/* Log lines */}
                        <div className="flex-1 overflow-y-auto bg-gray-950 rounded-b-lg font-mono text-xs">
                            {job.logs && job.logs.length > 0 ? (
                                <table className="w-full border-collapse">
                                    <tbody>
                                        {job.logs.map((raw, i) => {
                                            const parsed = parseLogLine(raw);
                                            const s = LOG_LEVEL_STYLES[parsed.level];
                                            return (
                                                <tr key={i} className={`${s.row} transition-colors`}>
                                                    <td className={`select-none pl-3 pr-2 py-0.5 text-right align-top w-10 shrink-0 ${s.lineNo} border-r border-gray-800`}>
                                                        {i + 1}
                                                    </td>
                                                    <td className={`pl-3 pr-3 py-0.5 align-top text-gray-300 ${wrapLines ? 'whitespace-pre-wrap break-all' : 'whitespace-pre'}`}>
                                                        {parsed.prefix && (
                                                            <span className={`${s.prefix} mr-1`}>{parsed.prefix}:</span>
                                                        )}
                                                        <LogMessageContent message={parsed.message || raw} />
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            ) : (
                                <div className="flex items-center justify-center h-32 text-gray-500 italic text-xs">
                                    No logs available yet...
                                </div>
                            )}
                            <div ref={logsEndRef} />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
