import React, { useState, useRef, useMemo, useEffect, useCallback } from 'react';
import { WrapText, ChevronsDown, ChevronsUp, CheckCircle, Copy } from 'lucide-react';
import { JobInfo } from '../../../../core/api/jobs';

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
    const level = pythonLogLevel(pyLogging[1]!);
    return { level, prefix: `${pyLogging[1]}:${pyLogging[2]}`, message: pyLogging[3] ?? '', raw };
  }
  // Match bracket style: "[INFO]", "[ERROR]", "[WARNING]"
  const bracket = /^\[(INFO|WARNING|WARN|ERROR|DEBUG)\]\s*(.*)/i.exec(raw);
  if (bracket) {
    const level = bracketLogLevel(bracket[1]!);
    return { level, prefix: `[${bracket[1]}]`, message: bracket[2] ?? '', raw };
  }
  // Match lines containing keywords
  const lower = raw.toLowerCase();
  if (/\b(error|exception|traceback|failed)\b/.test(lower)) return { level: 'error', prefix: '', message: raw, raw };
  if (/\b(warning|warn)\b/.test(lower)) return { level: 'warning', prefix: '', message: raw, raw };
  return { level: 'plain', prefix: '', message: raw, raw };
}

const LOG_LEVEL_STYLES: Record<LogLevel, { row: string; prefix: string; lineNo: string }> = {
  error: { row: 'bg-red-950/40 hover:bg-red-950/60', prefix: 'text-red-400 font-bold', lineNo: 'text-red-600/60' },
  warning: { row: 'bg-yellow-950/30 hover:bg-yellow-950/50', prefix: 'text-yellow-400 font-bold', lineNo: 'text-yellow-600/60' },
  info: { row: 'hover:bg-gray-800/50', prefix: 'text-blue-400', lineNo: 'text-gray-600' },
  debug: { row: 'hover:bg-gray-800/30', prefix: 'text-gray-500', lineNo: 'text-gray-700' },
  plain: { row: 'hover:bg-gray-800/40', prefix: '', lineNo: 'text-gray-600' },
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
      { text: '=', cls: 'text-gray-600' },
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
      { text: '=', cls: 'text-gray-600' },
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
  // Numeric rules skip starts inside a digit run to avoid quadratic retries on missing suffixes.
  // Keep the guard after -? so a minus following a digit still belongs to the token.
  // durations with units  e.g.  3.14s  250ms  0.05s
  {
    re: /-?(?<!\d)\d+(?:\.\d*)?(?:[eE][+-]?\d+)?\s*(?:ms|s)\b/,
    seg: m => [{ text: m[0], cls: 'text-teal-400' }],
  },
  // percentages  e.g.  98.5%
  {
    re: /-?(?<!\d)\d+(?:\.\d*)?%/,
    seg: m => [{ text: m[0], cls: 'text-emerald-400' }],
  },
  // floats (bare)
  {
    re: /-?(?<!\d)\d+\.\d+(?:[eE][+-]?\d+)?/,
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

function pythonLogLevel(value: string): LogLevel {
  const level = value.toUpperCase();
  if (level === 'ERROR' || level === 'CRITICAL') return 'error';
  if (level === 'WARNING') return 'warning';
  return level === 'DEBUG' ? 'debug' : 'info';
}

function bracketLogLevel(value: string): LogLevel {
  const level = value.toUpperCase();
  if (level === 'ERROR') return 'error';
  if (level === 'WARNING' || level === 'WARN') return 'warning';
  return level === 'DEBUG' ? 'debug' : 'info';
}

export function useJobLogs(job: JobInfo, activeTab: 'overview' | 'logs') {
  const [autoScroll, setAutoScroll] = useState(true);
  const [wrapLines, setWrapLines] = useState(true);
  const [copied, setCopied] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
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
  return { autoScroll, setAutoScroll, wrapLines, setWrapLines, copied, logsEndRef, handleCopyLogs };
}

type LogControls = ReturnType<typeof useJobLogs>;
function JobLogToolbar({ job, controls }: { job: JobInfo; controls: LogControls }) {
  return (
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
      <JobLogControls controls={controls} />
    </div>
  );
}

export function JobLogs({ job, controls }: { job: JobInfo; controls: LogControls }) {
  const { wrapLines, logsEndRef } = controls;
  return (
    <div className="flex flex-col h-full min-h-[400px]">
      {/* Log toolbar */}
      <JobLogToolbar job={job} controls={controls} />

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
  );
}

function JobLogControls({ controls }: { controls: LogControls }) {
  const { autoScroll, setAutoScroll, wrapLines, setWrapLines, copied, handleCopyLogs } = controls;
  return (
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
  );
}
