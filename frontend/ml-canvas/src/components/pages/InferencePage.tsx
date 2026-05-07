import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
    AlertCircle,
    BarChart3,
    Box,
    CheckCircle,
    Copy,
    Download,
    FileSpreadsheet,
    History,
    LayoutGrid,
    List,
    Play,
    Power,
    RotateCcw,
    Sparkles,
    Trash2,
    Upload,
    Wand2,
    X,
    Zap,
} from 'lucide-react';
import { deploymentApi, DeploymentInfo } from '../../core/api/deployment';
import { jobsApi } from '../../core/api/jobs';
import { DatasetService } from '../../core/api/datasets';
import { useConfirm } from '../shared';
import { toast } from '../../core/toast';

const DEFAULT_INPUT = '[\n  {\n    "feature1": 0.5,\n    "feature2": 1.2\n  }\n]';
const MAX_RECENT_RUNS = 5;
const SAMPLE_OPTIONS: ReadonlyArray<number> = [1, 5, 10, 25, 100];
const HISTOGRAM_BINS = 10;
const LARGE_BATCH_THRESHOLD = 500;

const LS_INPUT = 'inferencePage:lastInput';
const LS_SAMPLE_SIZE = 'inferencePage:sampleSize';
const LS_VIEW = 'inferencePage:resultsView';

/** Live status of the JSON typed into the textarea. */
interface InputStatus {
    valid: boolean;
    rows: number;
    message: string;
    /** 1-based line number when JSON parse fails, if extractable. */
    line?: number;
    column?: number;
}

/** Schema-vs-input drift summary surfaced under the editor. */
interface SchemaCheck {
    missing: string[]; // schema fields that no row provides
    extra: string[]; // row fields that the schema does not declare
    rows: number;
}

/** A previous prediction run kept in memory so the user can re-load it. */
interface RecentRun {
    at: number;
    rows: number;
    latencyMs: number;
    input: string;
    predictions: unknown[];
}

/** Try to extract a (line, column) tuple from a JSON parse error message. */
const extractJsonErrorPosition = (
    msg: string,
    raw: string,
): { line: number; column: number } | null => {
    // Chrome/V8: "Unexpected token } in JSON at position 42"
    const posMatch = /position\s+(\d+)/i.exec(msg);
    if (posMatch && posMatch[1]) {
        const pos = Number(posMatch[1]);
        if (!Number.isFinite(pos)) return null;
        const upTo = raw.slice(0, pos);
        const line = upTo.split('\n').length;
        const lastNl = upTo.lastIndexOf('\n');
        const column = lastNl === -1 ? pos + 1 : pos - lastNl;
        return { line, column };
    }
    // Firefox: "JSON.parse: ... at line 3 column 5 of the JSON data"
    const lineMatch = /line\s+(\d+)\s+column\s+(\d+)/i.exec(msg);
    if (lineMatch && lineMatch[1] && lineMatch[2]) {
        return { line: Number(lineMatch[1]), column: Number(lineMatch[2]) };
    }
    return null;
};

/** Inspect the textarea contents and produce a small live-status summary. */
const analyseInput = (raw: string): InputStatus => {
    try {
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) {
            return { valid: false, rows: 0, message: 'Input must be a JSON array of objects' };
        }
        return {
            valid: true,
            rows: parsed.length,
            message: `${parsed.length} row${parsed.length === 1 ? '' : 's'}`,
        };
    } catch (err) {
        const msg = (err as Error).message;
        const pos = extractJsonErrorPosition(msg, raw);
        if (pos) {
            return {
                valid: false,
                rows: 0,
                message: `${msg} (line ${pos.line}, col ${pos.column})`,
                line: pos.line,
                column: pos.column,
            };
        }
        return { valid: false, rows: 0, message: msg };
    }
};

/** Compare the current JSON input against the deployment schema. */
const checkSchema = (raw: string, schema: { name: string; type: string }[]): SchemaCheck | null => {
    if (schema.length === 0) return null;
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        return null;
    }
    if (!Array.isArray(parsed) || parsed.length === 0) return null;

    const schemaNames = new Set(schema.map(c => c.name));
    const seen = new Set<string>();
    parsed.forEach(row => {
        if (row && typeof row === 'object') {
            Object.keys(row as Record<string, unknown>).forEach(k => seen.add(k));
        }
    });
    const missing = [...schemaNames].filter(n => !seen.has(n));
    const extra = [...seen].filter(n => !schemaNames.has(n));
    return { missing, extra, rows: parsed.length };
};

/**
 * Numeric-looking cell? Accepts ints, decimals, scientific notation, signed.
 * The earlier `raw === String(num)` round-trip rejected `"1.0"` / `"1e5"` and
 * left them as strings, which then crashed the model with
 * `unsupported operand type(s) for -: 'str' and 'float'`.
 */
const NUMERIC_RE = /^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

/**
 * Tiny CSV-to-objects parser: comma-separated, first row is headers, no
 * quoted-field handling. Numeric-looking cells are coerced to numbers.
 */
const parseCsv = (text: string): Record<string, unknown>[] => {
    const lines = text.split(/\r?\n/).filter(l => l.trim() !== '');
    if (lines.length < 2) return [];
    const headers = (lines[0] ?? '').split(',').map(h => h.trim());
    return lines.slice(1).map(line => {
        const cells = line.split(',');
        const row: Record<string, unknown> = {};
        headers.forEach((h, i) => {
            const raw = (cells[i] ?? '').trim();
            if (raw === '') {
                row[h] = '';
                return;
            }
            if (NUMERIC_RE.test(raw)) {
                const num = Number(raw);
                row[h] = Number.isFinite(num) ? num : raw;
            } else {
                row[h] = raw;
            }
        });
        return row;
    });
};

/** Pull the predictions list down to plain numbers for stats display. */
const toNumericArray = (preds: unknown[]): number[] =>
    preds
        .map(p => (typeof p === 'number' ? p : Number(p)))
        .filter(n => Number.isFinite(n));

/** Format an epoch ms as HH:MM:SS for the recent-runs strip. */
const formatTime = (ms: number): string =>
    new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

/** Best-effort short string rendering for a single prediction value. */
const renderPrediction = (pred: unknown): string =>
    typeof pred === 'object' && pred !== null ? JSON.stringify(pred) : String(pred);

/** Detect classification probability shape: {classLabel: number, ...}. */
const asProbabilityMap = (pred: unknown): Record<string, number> | null => {
    if (!pred || typeof pred !== 'object' || Array.isArray(pred)) return null;
    const obj = pred as Record<string, unknown>;
    const entries = Object.entries(obj);
    if (entries.length === 0) return null;
    const numeric: Record<string, number> = {};
    for (const [k, v] of entries) {
        if (typeof v !== 'number' || !Number.isFinite(v)) return null;
        numeric[k] = v;
    }
    // Heuristic: at least one value in [0, 1] suggests probabilities.
    const vals = Object.values(numeric);
    if (!vals.some(v => v >= 0 && v <= 1)) return null;
    return numeric;
};

/**
 * Project a sample row onto the model's expected feature set: prefer the
 * deployment schema (with zero-fill for missing keys); otherwise fall back
 * to the raw row minus the known target / dropped columns.
 */
const projectSampleRow = (
    row: Record<string, unknown>,
    schema: { name: string; type: string }[],
    excluded: ReadonlySet<string>,
): Record<string, unknown> => {
    if (schema.length > 0) {
        const projected: Record<string, unknown> = {};
        schema.forEach(col => {
            if (excluded.has(col.name)) return;
            projected[col.name] = col.name in row ? row[col.name] : 0;
        });
        return projected;
    }
    const cleaned: Record<string, unknown> = {};
    Object.entries(row).forEach(([k, v]) => {
        if (!excluded.has(k)) cleaned[k] = v;
    });
    return cleaned;
};

/** Convert an array of rows into a CSV string for the export button. */
const rowsToCsv = (rows: Record<string, unknown>[]): string => {
    if (rows.length === 0) return '';
    const keys = Array.from(
        rows.reduce<Set<string>>((set, r) => {
            Object.keys(r).forEach(k => set.add(k));
            return set;
        }, new Set()),
    );
    const escape = (v: unknown): string => {
        const s = v == null ? '' : String(v);
        return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const header = keys.join(',');
    const body = rows.map(r => keys.map(k => escape(r[k])).join(',')).join('\n');
    return `${header}\n${body}`;
};

/** Tiny inline SVG histogram of numeric predictions — no recharts dep. */
const PredictionHistogram: React.FC<{ values: number[] }> = ({ values }) => {
    const stats = useMemo(() => {
        if (values.length === 0) return null;
        const min = Math.min(...values);
        const max = Math.max(...values);
        if (min === max) return { min, max, bins: [values.length] as number[] };
        const binWidth = (max - min) / HISTOGRAM_BINS;
        const bins = new Array<number>(HISTOGRAM_BINS).fill(0);
        values.forEach(v => {
            let idx = Math.floor((v - min) / binWidth);
            if (idx >= HISTOGRAM_BINS) idx = HISTOGRAM_BINS - 1;
            const safe = bins[idx];
            bins[idx] = (safe ?? 0) + 1;
        });
        return { min, max, bins };
    }, [values]);

    if (!stats) return null;
    const width = 220;
    const height = 48;
    const peak = Math.max(...stats.bins, 1);
    const barWidth = width / stats.bins.length;

    return (
        <div className="flex items-center gap-2" title="Distribution of predictions">
            <svg width={width} height={height} className="text-blue-500">
                {stats.bins.map((count, i) => {
                    const h = (count / peak) * (height - 4);
                    return (
                        <rect
                            key={i}
                            x={i * barWidth + 1}
                            y={height - h}
                            width={Math.max(1, barWidth - 2)}
                            height={h}
                            fill="currentColor"
                            opacity={0.85}
                        />
                    );
                })}
            </svg>
            <div className="flex flex-col text-[10px] text-gray-400 leading-tight">
                <span className="tabular-nums">{stats.max.toFixed(2)}</span>
                <span className="tabular-nums">{stats.min.toFixed(2)}</span>
            </div>
        </div>
    );
};

/** Sparkline of recent-run latencies — surfaced in the "Recent runs" header. */
const LatencySparkline: React.FC<{ values: number[] }> = ({ values }) => {
    if (values.length < 2) return null;
    const width = 60;
    const height = 16;
    const max = Math.max(...values);
    const min = Math.min(...values);
    const range = max - min || 1;
    const points = values
        .map((v, i) => {
            const x = (i / (values.length - 1)) * width;
            const y = height - ((v - min) / range) * (height - 2) - 1;
            return `${x},${y}`;
        })
        .join(' ');
    return (
        <svg width={width} height={height} className="text-blue-400" aria-hidden="true">
            <polyline
                points={points}
                fill="none"
                stroke="currentColor"
                strokeWidth="1.25"
                strokeLinejoin="round"
            />
        </svg>
    );
};

/** Bar visualisation for a {class: probability} prediction. */
const ProbabilityBars: React.FC<{ probs: Record<string, number> }> = ({ probs }) => {
    const entries = Object.entries(probs).sort(([, a], [, b]) => b - a);
    const peak = Math.max(...entries.map(([, v]) => v), 1);
    return (
        <div className="flex flex-col gap-1 w-full">
            {entries.map(([cls, p], i) => {
                const pct = (p / peak) * 100;
                const isTop = i === 0;
                return (
                    <div key={cls} className="flex items-center gap-2 text-[11px]">
                        <span
                            className={`font-mono shrink-0 w-20 truncate ${
                                isTop
                                    ? 'text-blue-700 dark:text-blue-300 font-semibold'
                                    : 'text-gray-500 dark:text-gray-400'
                            }`}
                            title={cls}
                        >
                            {cls}
                        </span>
                        <div className="flex-1 h-2 bg-gray-100 dark:bg-gray-700/50 rounded overflow-hidden">
                            <div
                                className={`h-full ${
                                    isTop ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
                                }`}
                                style={{ width: `${pct}%` }}
                            />
                        </div>
                        <span className="tabular-nums w-12 text-right text-gray-600 dark:text-gray-300">
                            {(p * 100).toFixed(1)}%
                        </span>
                    </div>
                );
            })}
        </div>
    );
};

/** Side-by-side table view: each input row beside its prediction. */
const InputOutputTable: React.FC<{ rows: unknown[]; predictions: unknown[] }> = ({
    rows,
    predictions,
}) => {
    const keys = useMemo(() => {
        const set = new Set<string>();
        rows.forEach(r => {
            if (r && typeof r === 'object') {
                Object.keys(r as Record<string, unknown>).forEach(k => set.add(k));
            }
        });
        return [...set];
    }, [rows]);

    if (rows.length === 0) return null;

    return (
        <div className="overflow-auto h-full">
            <table className="w-full text-xs border-collapse">
                <thead className="sticky top-0 bg-gray-50 dark:bg-gray-900">
                    <tr>
                        <th className="text-left px-2 py-1 text-gray-400 font-medium border-b border-gray-200 dark:border-gray-700 w-8">
                            #
                        </th>
                        {keys.map(k => (
                            <th
                                key={k}
                                className="text-left px-2 py-1 text-gray-500 dark:text-gray-400 font-medium border-b border-gray-200 dark:border-gray-700"
                            >
                                {k}
                            </th>
                        ))}
                        <th className="text-left px-2 py-1 text-blue-500 font-semibold border-b border-gray-200 dark:border-gray-700 sticky right-0 bg-gray-50 dark:bg-gray-900">
                            prediction
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, i) => {
                        const obj = (row && typeof row === 'object'
                            ? (row as Record<string, unknown>)
                            : {}) as Record<string, unknown>;
                        return (
                            <tr
                                key={i}
                                className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/40"
                            >
                                <td className="px-2 py-1 text-gray-400 tabular-nums">{i + 1}</td>
                                {keys.map(k => (
                                    <td
                                        key={k}
                                        className="px-2 py-1 font-mono text-gray-700 dark:text-gray-300 truncate max-w-[140px]"
                                        title={String(obj[k] ?? '')}
                                    >
                                        {obj[k] === undefined ? (
                                            <span className="text-gray-300 dark:text-gray-600">—</span>
                                        ) : (
                                            String(obj[k])
                                        )}
                                    </td>
                                ))}
                                <td className="px-2 py-1 font-mono font-medium text-blue-600 dark:text-blue-400 sticky right-0 bg-white dark:bg-gray-800">
                                    {renderPrediction(predictions[i])}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
};

/** Polished segmented control for picking the sample-row count. */
const SampleSizeSegmented: React.FC<{
    value: number;
    onChange: (n: number) => void;
    disabled: boolean;
}> = ({ value, onChange, disabled }) => (
    <div
        className={`inline-flex rounded-md overflow-hidden border ${
            disabled
                ? 'border-gray-200 dark:border-gray-700 opacity-50 cursor-not-allowed'
                : 'border-gray-200 dark:border-gray-600'
        }`}
        role="group"
        aria-label="Sample row count"
    >
        {SAMPLE_OPTIONS.map(n => {
            const active = n === value;
            return (
                <button
                    key={n}
                    type="button"
                    disabled={disabled}
                    onClick={() => onChange(n)}
                    className={`px-2 py-1 text-[11px] font-medium tabular-nums transition-colors border-r last:border-r-0 ${
                        disabled ? 'border-gray-200 dark:border-gray-700' : 'border-gray-200 dark:border-gray-600'
                    } ${
                        active
                            ? 'bg-blue-600 text-white'
                            : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-blue-900/30'
                    }`}
                    title={`Fetch ${n} sample row${n === 1 ? '' : 's'}`}
                >
                    {n}
                </button>
            );
        })}
    </div>
);

export const InferencePage: React.FC = () => {
    const confirm = useConfirm();
    const csvInputRef = useRef<HTMLInputElement>(null);
    const editorWrapRef = useRef<HTMLDivElement>(null);

    const [activeDeployment, setActiveDeployment] = useState<DeploymentInfo | null>(null);
    const [datasetId, setDatasetId] = useState<string | null>(null);
    /** Columns the model never trained on (target + user-dropped). */
    const [excludedColumns, setExcludedColumns] = useState<Set<string>>(new Set());
    const [inputData, setInputData] = useState<string>(() => {
        try {
            return localStorage.getItem(LS_INPUT) ?? DEFAULT_INPUT;
        } catch {
            return DEFAULT_INPUT;
        }
    });
    const [predictions, setPredictions] = useState<unknown[] | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isReloadingSample, setIsReloadingSample] = useState(false);
    const [sampleSize, setSampleSize] = useState<number>(() => {
        try {
            const v = Number(localStorage.getItem(LS_SAMPLE_SIZE));
            return SAMPLE_OPTIONS.includes(v) ? v : 1;
        } catch {
            return 1;
        }
    });
    const [error, setError] = useState<string | null>(null);
    const [latencyMs, setLatencyMs] = useState<number | null>(null);

    const [autoFilterInfo, setAutoFilterInfo] = useState<string | null>(null);
    const [bannerDismissed, setBannerDismissed] = useState(false);

    const [recentRuns, setRecentRuns] = useState<RecentRun[]>([]);
    const [resultsView, setResultsView] = useState<'list' | 'table'>(() => {
        try {
            const v = localStorage.getItem(LS_VIEW);
            return v === 'table' ? 'table' : 'list';
        } catch {
            return 'list';
        }
    });
    const [isDragging, setIsDragging] = useState(false);

    const inputStatus = useMemo(() => analyseInput(inputData), [inputData]);

    const schemaChips = useMemo(
        () => activeDeployment?.input_schema ?? [],
        [activeDeployment],
    );

    const schemaCheck = useMemo(
        () => checkSchema(inputData, schemaChips),
        [inputData, schemaChips],
    );

    const predictionStats = useMemo(() => {
        if (!predictions || predictions.length === 0) return null;
        const nums = toNumericArray(predictions);
        if (nums.length === 0) return null;
        const min = Math.min(...nums);
        const max = Math.max(...nums);
        const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
        return { count: nums.length, mean, min, max, values: nums };
    }, [predictions]);

    const parsedInputRows = useMemo(() => {
        if (!inputStatus.valid) return [];
        try {
            const arr = JSON.parse(inputData);
            return Array.isArray(arr) ? (arr as unknown[]) : [];
        } catch {
            return [];
        }
    }, [inputStatus.valid, inputData]);

    /** Detect "this is a single classification probability response" shape. */
    const singleProbMap = useMemo(() => {
        if (!predictions || predictions.length !== 1) return null;
        return asProbabilityMap(predictions[0]);
    }, [predictions]);

    /** Persist user preferences across reloads. */
    useEffect(() => {
        try {
            localStorage.setItem(LS_INPUT, inputData);
        } catch {
            /* storage may be disabled in private mode — ignore */
        }
    }, [inputData]);
    useEffect(() => {
        try {
            localStorage.setItem(LS_SAMPLE_SIZE, String(sampleSize));
        } catch {
            /* ignore */
        }
    }, [sampleSize]);
    useEffect(() => {
        try {
            localStorage.setItem(LS_VIEW, resultsView);
        } catch {
            /* ignore */
        }
    }, [resultsView]);

    const loadActiveDeployment = useCallback(async () => {
        try {
            const deployment = await deploymentApi.getActive();
            setActiveDeployment(deployment);
            if (!deployment) {
                setDatasetId(null);
                setExcludedColumns(new Set());
                return;
            }

            let initialData: Record<string, unknown> = {};
            let usedSchema = false;
            const excluded = new Set<string>();

            if (deployment.input_schema && deployment.input_schema.length > 0) {
                deployment.input_schema.forEach(col => {
                    initialData[col.name] = 0;
                });
                usedSchema = true;
                setAutoFilterInfo(
                    `Schema loaded from artifacts (${deployment.input_schema.length} features)`,
                );
            }

            if (deployment.job_id) {
                try {
                    const job = await jobsApi.getJob(deployment.job_id);
                    const targetColumn = job.target_column;
                    const droppedColumns = job.dropped_columns || [];
                    if (typeof targetColumn === 'string' && targetColumn) excluded.add(targetColumn);
                    droppedColumns.forEach(col => {
                        if (typeof col === 'string' && col) excluded.add(col);
                    });

                    if (job.dataset_id) {
                        setDatasetId(job.dataset_id);
                        const sample = await DatasetService.getSample(job.dataset_id, 1);
                        if (sample.length > 0) {
                            const sampleRow = sample[0] as Record<string, unknown>;

                            if (usedSchema) {
                                Object.keys(initialData).forEach(key => {
                                    if (key in sampleRow) initialData[key] = sampleRow[key];
                                });
                            } else {
                                initialData = projectSampleRow(sampleRow, [], excluded);

                                const droppedInfo: string[] = [];
                                if (targetColumn) droppedInfo.push(`Target: ${targetColumn}`);
                                if (droppedColumns.length > 0) {
                                    droppedInfo.push(`Dropped: ${droppedColumns.length} cols`);
                                }
                                if (droppedInfo.length > 0) {
                                    setAutoFilterInfo(
                                        `Auto-filtered from dataset: ${droppedInfo.join(', ')}`,
                                    );
                                }
                            }
                        }
                    }
                } catch (err) {
                    console.warn('Failed to fetch dataset sample', err);
                }
            }

            setExcludedColumns(excluded);

            // Only seed the editor with a fresh sample when there is no
            // user-restored input from localStorage (so reloads keep state).
            if (Object.keys(initialData).length > 0) {
                let userInputIsDefault = false;
                try {
                    const stored = localStorage.getItem(LS_INPUT);
                    userInputIsDefault = stored == null || stored === DEFAULT_INPUT;
                } catch {
                    userInputIsDefault = true;
                }
                if (userInputIsDefault) {
                    setInputData(JSON.stringify([initialData], null, 2));
                }
            }
        } catch (e) {
            console.error('Failed to load active deployment', e);
            setActiveDeployment(null);
            setDatasetId(null);
            setExcludedColumns(new Set());
        }
    }, []);

    useEffect(() => {
        void loadActiveDeployment();
    }, [loadActiveDeployment]);

    const handleReloadSample = useCallback(async () => {
        if (!datasetId || isReloadingSample) return;
        setIsReloadingSample(true);
        try {
            const sample = await DatasetService.getSample(datasetId, sampleSize);
            if (sample.length === 0) {
                toast.error('Dataset returned no rows');
                return;
            }
            const rows = sample.map(r =>
                projectSampleRow(
                    r as Record<string, unknown>,
                    schemaChips,
                    excludedColumns,
                ),
            );
            setInputData(JSON.stringify(rows, null, 2));
            toast.success(
                `Loaded ${rows.length} sample row${rows.length === 1 ? '' : 's'}` +
                    (excludedColumns.size > 0 ? ` (excluded ${excludedColumns.size} cols)` : ''),
            );
        } catch (e) {
            console.error('Failed to reload sample', e);
            toast.error('Could not fetch new samples');
        } finally {
            setIsReloadingSample(false);
        }
    }, [datasetId, isReloadingSample, sampleSize, schemaChips, excludedColumns]);

    /** Shared CSV-text → JSON-array writer (used by file picker and DnD). */
    const ingestCsvText = useCallback(
        (text: string) => {
            try {
                const parsed = parseCsv(text);
                if (parsed.length === 0) {
                    toast.error('CSV had no data rows');
                    return;
                }
                const rows = parsed.map(r => {
                    if (excludedColumns.size === 0) return r;
                    const cleaned: Record<string, unknown> = {};
                    Object.entries(r).forEach(([k, v]) => {
                        if (!excludedColumns.has(k)) cleaned[k] = v;
                    });
                    return cleaned;
                });
                setInputData(JSON.stringify(rows, null, 2));
                toast.success(`Loaded ${rows.length} row${rows.length === 1 ? '' : 's'} from CSV`);
            } catch (e) {
                console.error('CSV parse failed', e);
                toast.error('Could not parse CSV');
            }
        },
        [excludedColumns],
    );

    const handleCsvFile = useCallback(
        (file: File) => {
            const reader = new FileReader();
            reader.onload = () => ingestCsvText(String(reader.result ?? ''));
            reader.onerror = () => toast.error('Could not read file');
            reader.readAsText(file);
        },
        [ingestCsvText],
    );

    const handleCsvChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) handleCsvFile(file);
        e.target.value = '';
    };

    /** Drag-and-drop CSV onto the editor. */
    const handleDragOver = (e: React.DragEvent) => {
        if (e.dataTransfer.types.includes('Files')) {
            e.preventDefault();
            setIsDragging(true);
        }
    };
    const handleDragLeave = (e: React.DragEvent) => {
        // Only clear when leaving the wrapper itself, not its children.
        if (e.currentTarget === e.target) setIsDragging(false);
    };
    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (!file) return;
        if (!file.name.toLowerCase().endsWith('.csv')) {
            toast.error('Only .csv files are supported');
            return;
        }
        handleCsvFile(file);
    };

    /** Pad missing schema fields with zero in every row of the input. */
    const handleFixMissingFields = () => {
        if (!schemaCheck || schemaCheck.missing.length === 0) return;
        try {
            const arr = JSON.parse(inputData) as Record<string, unknown>[];
            const padded = arr.map(row => {
                const next = { ...row };
                schemaCheck.missing.forEach(field => {
                    if (!(field in next)) next[field] = 0;
                });
                return next;
            });
            setInputData(JSON.stringify(padded, null, 2));
            toast.success(`Added ${schemaCheck.missing.length} missing field(s)`);
        } catch {
            toast.error('Cannot pad: invalid JSON');
        }
    };

    const handleDeactivate = async () => {
        const ok = await confirm({
            title: 'Undeploy model?',
            message: 'Are you sure you want to undeploy the current model?',
            confirmLabel: 'Undeploy',
            variant: 'danger',
        });
        if (!ok) return;
        try {
            await deploymentApi.deactivate();
            setActiveDeployment(null);
            setDatasetId(null);
            setExcludedColumns(new Set());
            setPredictions(null);
            setLatencyMs(null);
            setRecentRuns([]);
        } catch (e) {
            console.error('Failed to deactivate', e);
            toast.error('Failed to undeploy model');
        }
    };

    const handlePredict = useCallback(async () => {
        if (!activeDeployment || isLoading) return;

        // Soft warning before sending huge payloads — the network round-trip
        // and JSON serialisation balloon, and it's almost always a mistake.
        let data: unknown;
        try {
            data = JSON.parse(inputData);
        } catch (e) {
            setError((e as Error).message);
            return;
        }
        if (!Array.isArray(data)) {
            setError('Input must be a JSON array of objects');
            return;
        }
        if (data.length > LARGE_BATCH_THRESHOLD) {
            const ok = await confirm({
                title: 'Large batch',
                message: `You are about to send ${data.length} rows. Continue?`,
                confirmLabel: 'Send',
            });
            if (!ok) return;
        }

        setIsLoading(true);
        setError(null);
        setPredictions(null);
        setLatencyMs(null);
        const start = performance.now();
        try {
            const response = await deploymentApi.predict(data);
            const elapsed = Math.round(performance.now() - start);
            setPredictions(response.predictions);
            setLatencyMs(elapsed);
            setRecentRuns(prev =>
                [
                    {
                        at: Date.now(),
                        rows: data.length,
                        latencyMs: elapsed,
                        input: inputData,
                        predictions: response.predictions,
                    },
                    ...prev,
                ].slice(0, MAX_RECENT_RUNS),
            );
        } catch (e: unknown) {
            setError((e as Error).message || 'Prediction failed');
        } finally {
            setIsLoading(false);
        }
    }, [activeDeployment, inputData, isLoading, confirm]);

    const handleFormatJson = () => {
        try {
            const parsed = JSON.parse(inputData);
            setInputData(JSON.stringify(parsed, null, 2));
        } catch {
            toast.error('Cannot format: invalid JSON');
        }
    };

    /** Wipe input + results back to the empty default. */
    const handleClearInput = () => {
        setInputData(DEFAULT_INPUT);
        setPredictions(null);
        setError(null);
        setLatencyMs(null);
    };

    const handleTextareaKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            e.preventDefault();
            void handlePredict();
        }
    };

    const handleCopyPredictions = async () => {
        if (!predictions) return;
        try {
            await navigator.clipboard.writeText(JSON.stringify(predictions, null, 2));
            toast.success('Copied predictions to clipboard');
        } catch {
            toast.error('Failed to copy');
        }
    };

    /** Trigger a file download for the given Blob. */
    const triggerDownload = (blob: Blob, filename: string) => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const handleDownloadJson = () => {
        if (!predictions) return;
        triggerDownload(
            new Blob([JSON.stringify(predictions, null, 2)], { type: 'application/json' }),
            `predictions_${Date.now()}.json`,
        );
    };

    /** Export inputs + predictions side-by-side to CSV. */
    const handleDownloadCsv = () => {
        if (!predictions) return;
        const merged: Record<string, unknown>[] = parsedInputRows.map((row, i) => {
            const obj = (row && typeof row === 'object'
                ? { ...(row as Record<string, unknown>) }
                : {}) as Record<string, unknown>;
            obj.prediction = renderPrediction(predictions[i]);
            return obj;
        });
        // Fallback: predictions-only column when there are no parsed input rows.
        const rows =
            merged.length > 0
                ? merged
                : predictions.map((p, i) => ({ row: i + 1, prediction: renderPrediction(p) }));
        triggerDownload(
            new Blob([rowsToCsv(rows)], { type: 'text/csv;charset=utf-8' }),
            `predictions_${Date.now()}.csv`,
        );
    };

    const handleRestoreRun = (run: RecentRun) => {
        setInputData(run.input);
        setPredictions(run.predictions);
        setLatencyMs(run.latencyMs);
        setError(null);
    };

    const showBanner = autoFilterInfo && !bannerDismissed;
    const hasSchemaIssues =
        schemaCheck && (schemaCheck.missing.length > 0 || schemaCheck.extra.length > 0);
    const recentLatencies = useMemo(
        () => [...recentRuns].reverse().map(r => r.latencyMs),
        [recentRuns],
    );

    return (
        <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 p-6 overflow-hidden">
            <h1 className="text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-6 shrink-0 flex items-center gap-2">
                <Zap className="w-6 h-6 text-blue-500" />
                Testing Model Inference
            </h1>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
                {/* Left column: input editor */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col h-full min-h-0">
                    <div className="flex justify-between items-start mb-3 gap-3 flex-wrap">
                        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">
                            Input Data (JSON)
                        </h3>
                        <div className="flex items-center gap-1 flex-wrap">
                            <input
                                ref={csvInputRef}
                                type="file"
                                accept=".csv,text/csv"
                                className="hidden"
                                onChange={handleCsvChange}
                            />
                            <button
                                onClick={() => csvInputRef.current?.click()}
                                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                                title="Upload a CSV file (or drag & drop). Target/dropped columns are stripped automatically."
                            >
                                <Upload className="w-3 h-3" /> CSV
                            </button>
                            <SampleSizeSegmented
                                value={sampleSize}
                                onChange={setSampleSize}
                                disabled={!datasetId}
                            />
                            <button
                                onClick={() => void handleReloadSample()}
                                disabled={!datasetId || isReloadingSample}
                                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed"
                                title="Load fresh random rows from the training dataset (target / dropped columns excluded)"
                            >
                                <RotateCcw
                                    className={`w-3 h-3 ${isReloadingSample ? 'animate-spin' : ''}`}
                                />
                                Sample
                            </button>
                            <button
                                onClick={handleFormatJson}
                                disabled={!inputStatus.valid}
                                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed"
                                title="Pretty-print JSON"
                            >
                                <Sparkles className="w-3 h-3" /> Format
                            </button>
                            <button
                                onClick={handleClearInput}
                                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                                title="Reset the editor and clear results"
                            >
                                <Trash2 className="w-3 h-3" /> Clear
                            </button>
                        </div>
                    </div>

                    {/* Schema chips: scrollable strip showing expected fields. */}
                    {schemaChips.length > 0 && (
                        <div className="mb-3 flex items-center gap-2 overflow-x-auto pb-1 scrollbar-thin">
                            <span className="text-[10px] uppercase tracking-wider text-gray-400 shrink-0">
                                Schema
                            </span>
                            {schemaChips.map(col => (
                                <span
                                    key={col.name}
                                    className="shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-600"
                                    title={`${col.name} (${col.type})`}
                                >
                                    <span className="font-mono">{col.name}</span>
                                    <span className="text-gray-400 dark:text-gray-500">{col.type}</span>
                                </span>
                            ))}
                        </div>
                    )}

                    {/* Excluded columns reminder — only if any are configured. */}
                    {excludedColumns.size > 0 && (
                        <div className="mb-3 flex items-center gap-2 overflow-x-auto pb-1 scrollbar-thin text-[11px]">
                            <span className="text-[10px] uppercase tracking-wider text-gray-400 shrink-0">
                                Excluded
                            </span>
                            {[...excludedColumns].map(col => (
                                <span
                                    key={col}
                                    className="shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-rose-50 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800"
                                    title="Stripped automatically from CSV uploads and Sample calls"
                                >
                                    <Trash2 className="w-2.5 h-2.5" />
                                    <span className="font-mono">{col}</span>
                                </span>
                            ))}
                        </div>
                    )}

                    {showBanner && (
                        <div className="mb-3 flex items-start gap-2 text-xs text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-900/20 px-3 py-2 rounded border border-amber-200 dark:border-amber-800">
                            <span className="flex-1">
                                {autoFilterInfo}
                                <span className="block text-[10px] text-amber-600/80 dark:text-amber-400/80 mt-0.5">
                                    Please verify fields before running.
                                </span>
                            </span>
                            <button
                                onClick={() => setBannerDismissed(true)}
                                className="text-amber-500 hover:text-amber-700 dark:hover:text-amber-200"
                                title="Dismiss"
                            >
                                <X className="w-3 h-3" />
                            </button>
                        </div>
                    )}

                    {/* Editor wrapper handles drag-and-drop CSV. */}
                    <div
                        ref={editorWrapRef}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className="relative flex-1 min-h-0"
                    >
                        <textarea
                            className="absolute inset-0 w-full h-full p-4 font-mono text-sm bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none resize-none"
                            value={inputData}
                            onChange={e => setInputData(e.target.value)}
                            onKeyDown={handleTextareaKeyDown}
                            placeholder='[{"col1": 1, "col2": "A"}]'
                            spellCheck={false}
                        />
                        {isDragging && (
                            <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-blue-500/10 dark:bg-blue-400/10 border-2 border-dashed border-blue-400 dark:border-blue-500 rounded-lg">
                                <div className="flex items-center gap-2 text-blue-600 dark:text-blue-300 font-medium text-sm">
                                    <Upload className="w-5 h-5" />
                                    Drop CSV to ingest
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Schema validation badges + one-click fix. */}
                    {hasSchemaIssues && (
                        <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px]">
                            {schemaCheck.missing.length > 0 && (
                                <>
                                    <span
                                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-rose-50 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800"
                                        title={schemaCheck.missing.join(', ')}
                                    >
                                        <AlertCircle className="w-3 h-3" />
                                        {schemaCheck.missing.length} missing
                                    </span>
                                    <button
                                        onClick={handleFixMissingFields}
                                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800 hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors"
                                        title="Add missing fields with value 0 in every row"
                                    >
                                        <Wand2 className="w-3 h-3" /> Fix
                                    </button>
                                </>
                            )}
                            {schemaCheck.extra.length > 0 && (
                                <span
                                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800"
                                    title={schemaCheck.extra.join(', ')}
                                >
                                    <AlertCircle className="w-3 h-3" />
                                    {schemaCheck.extra.length} extra
                                </span>
                            )}
                            <span className="text-[10px] text-gray-400 italic ml-1">
                                Hover badges for field names.
                            </span>
                        </div>
                    )}

                    <div className="mt-3 flex justify-between items-center gap-3 flex-wrap">
                        <span
                            className={`text-xs flex items-center gap-1 ${
                                inputStatus.valid
                                    ? 'text-gray-500 dark:text-gray-400'
                                    : 'text-red-600 dark:text-red-400'
                            }`}
                        >
                            {inputStatus.valid ? (
                                <>
                                    <CheckCircle className="w-3 h-3" /> {inputStatus.message}
                                </>
                            ) : (
                                <>
                                    <AlertCircle className="w-3 h-3" /> {inputStatus.message}
                                </>
                            )}
                        </span>
                        <div className="flex items-center gap-3">
                            <kbd className="hidden md:inline-flex items-center gap-1 text-[10px] text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded border border-gray-200 dark:border-gray-600 font-mono">
                                Ctrl
                                <span className="text-gray-400">+</span>
                                Enter
                            </kbd>
                            <button
                                onClick={() => void handlePredict()}
                                disabled={!activeDeployment || isLoading || !inputStatus.valid}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                                    !activeDeployment || isLoading || !inputStatus.valid
                                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed dark:bg-gray-800 dark:text-gray-600'
                                        : 'bg-blue-600 text-white hover:bg-blue-700 shadow-sm'
                                }`}
                            >
                                {isLoading ? (
                                    'Running...'
                                ) : (
                                    <>
                                        <Play className="w-4 h-4" /> Run Prediction
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Right column: deployment status + results */}
                <div className="flex flex-col gap-6 h-full min-h-0">
                    {/* Active deployment card */}
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 shrink-0">
                        <h2 className="text-lg font-medium text-gray-800 dark:text-gray-100 mb-4 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Box className="w-5 h-5 text-blue-500" />
                                Active Deployment
                            </div>
                            {activeDeployment && (
                                <button
                                    onClick={() => void handleDeactivate()}
                                    className="flex items-center gap-1 text-xs px-2 py-1 bg-red-100 text-red-600 rounded hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50 transition-colors"
                                    title="Undeploy model"
                                >
                                    <Power className="w-3 h-3" /> Undeploy
                                </button>
                            )}
                        </h2>
                        {activeDeployment ? (
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <div className="text-xs text-gray-500 dark:text-gray-400">Model Type</div>
                                    <div className="font-medium text-gray-800 dark:text-gray-200">
                                        {activeDeployment.model_type}
                                    </div>
                                </div>
                                <div>
                                    <div className="text-xs text-gray-500 dark:text-gray-400">Job ID</div>
                                    <div className="font-mono text-sm text-gray-800 dark:text-gray-200 break-all">
                                        {activeDeployment.job_id}
                                    </div>
                                </div>
                                <div>
                                    <div className="text-xs text-gray-500 dark:text-gray-400">Deployed At</div>
                                    <div className="text-sm text-gray-800 dark:text-gray-200">
                                        {new Date(activeDeployment.created_at).toLocaleDateString()}
                                    </div>
                                </div>
                                <div>
                                    <div className="text-xs text-gray-500 dark:text-gray-400">Status</div>
                                    <div className="flex items-center gap-1 text-green-600 dark:text-green-400 font-medium text-sm">
                                        <CheckCircle className="w-4 h-4" /> Active
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="flex flex-col gap-3">
                                <div className="text-gray-500 dark:text-gray-400 italic flex items-center gap-2">
                                    <AlertCircle className="w-4 h-4" />
                                    No model is currently deployed.
                                </div>
                                <Link
                                    to="/registry"
                                    className="self-start inline-flex items-center gap-2 text-xs px-3 py-1.5 bg-blue-50 text-blue-600 rounded hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-400 dark:hover:bg-blue-900/50 transition-colors"
                                >
                                    <Box className="w-3 h-3" /> Browse Model Registry
                                </Link>
                            </div>
                        )}
                    </div>

                    {/* Results panel */}
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col flex-1 min-h-0">
                        <div className="flex justify-between items-center mb-4 gap-3 flex-wrap">
                            <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">
                                Prediction Results
                            </h3>
                            <div className="flex items-center gap-3 flex-wrap">
                                {latencyMs != null && (
                                    <span className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                                        <Zap className="w-3 h-3" /> {latencyMs} ms
                                    </span>
                                )}
                                {predictions && (
                                    <>
                                        <div className="flex border border-gray-200 dark:border-gray-700 rounded overflow-hidden">
                                            <button
                                                onClick={() => setResultsView('list')}
                                                className={`flex items-center gap-1 text-xs px-2 py-1 transition-colors ${
                                                    resultsView === 'list'
                                                        ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300'
                                                        : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                                                }`}
                                                title="List view"
                                            >
                                                <List className="w-3 h-3" /> List
                                            </button>
                                            <button
                                                onClick={() => setResultsView('table')}
                                                className={`flex items-center gap-1 text-xs px-2 py-1 transition-colors border-l border-gray-200 dark:border-gray-700 ${
                                                    resultsView === 'table'
                                                        ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300'
                                                        : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                                                }`}
                                                title="Side-by-side input/output table"
                                            >
                                                <LayoutGrid className="w-3 h-3" /> Table
                                            </button>
                                        </div>
                                        <button
                                            onClick={() => void handleCopyPredictions()}
                                            className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                                            title="Copy as JSON"
                                        >
                                            <Copy className="w-3 h-3" /> Copy
                                        </button>
                                        <button
                                            onClick={handleDownloadJson}
                                            className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                                            title="Download as JSON"
                                        >
                                            <Download className="w-3 h-3" /> JSON
                                        </button>
                                        <button
                                            onClick={handleDownloadCsv}
                                            className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                                            title="Download inputs + predictions as CSV"
                                        >
                                            <FileSpreadsheet className="w-3 h-3" /> CSV
                                        </button>
                                    </>
                                )}
                            </div>
                        </div>

                        {/* Numeric stats strip + tiny histogram. */}
                        {predictionStats && (
                            <div className="mb-3 flex flex-wrap items-center gap-3">
                                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                                    <span className="inline-flex items-center gap-1 text-gray-400">
                                        <BarChart3 className="w-3 h-3" /> Stats
                                    </span>
                                    <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200">
                                        n = {predictionStats.count}
                                    </span>
                                    <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200 tabular-nums">
                                        mean {predictionStats.mean.toFixed(4)}
                                    </span>
                                    <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200 tabular-nums">
                                        min {predictionStats.min.toFixed(4)}
                                    </span>
                                    <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200 tabular-nums">
                                        max {predictionStats.max.toFixed(4)}
                                    </span>
                                </div>
                                {predictionStats.values.length > 1 && (
                                    <PredictionHistogram values={predictionStats.values} />
                                )}
                            </div>
                        )}

                        {/* Probability bars for single-row classification responses. */}
                        {singleProbMap && (
                            <div className="mb-3 p-3 rounded bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700">
                                <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-2 flex items-center gap-1">
                                    <BarChart3 className="w-3 h-3" /> Class probabilities
                                </div>
                                <ProbabilityBars probs={singleProbMap} />
                            </div>
                        )}

                        <div className="flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                            {error ? (
                                <div className="p-4 text-red-600 dark:text-red-400 flex items-start gap-2">
                                    <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                                    <pre className="whitespace-pre-wrap font-mono text-sm">{error}</pre>
                                </div>
                            ) : predictions ? (
                                resultsView === 'table' && parsedInputRows.length > 0 ? (
                                    <InputOutputTable
                                        rows={parsedInputRows}
                                        predictions={predictions}
                                    />
                                ) : (
                                    <div className="p-4 space-y-2 overflow-auto h-full">
                                        {predictions.map((pred, i) => {
                                            const probs = asProbabilityMap(pred);
                                            return (
                                                <div
                                                    key={i}
                                                    className="flex items-start gap-3 p-2 bg-white dark:bg-gray-800 rounded border border-gray-100 dark:border-gray-700"
                                                >
                                                    <span className="text-xs text-gray-500 w-8 shrink-0 mt-0.5">
                                                        #{i + 1}
                                                    </span>
                                                    {probs ? (
                                                        <div className="flex-1 min-w-0">
                                                            <ProbabilityBars probs={probs} />
                                                        </div>
                                                    ) : (
                                                        <span className="font-mono font-medium text-blue-600 dark:text-blue-400 break-all">
                                                            {renderPrediction(pred)}
                                                        </span>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                )
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center text-gray-400 text-sm italic gap-2 p-4 text-center">
                                    <Play className="w-8 h-8 opacity-40" />
                                    Run a prediction to see results here.
                                    {schemaChips.length > 0 && (
                                        <span className="text-[11px] not-italic text-gray-500 dark:text-gray-500">
                                            Tip: click <strong>Sample</strong> to load fresh rows from
                                            the training dataset.
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>

                        {/* Recent runs strip — click to restore an earlier run. */}
                        {recentRuns.length > 0 && (
                            <div className="mt-3 border-t border-gray-100 dark:border-gray-700 pt-3">
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-gray-400">
                                        <History className="w-3 h-3" /> Recent runs
                                    </div>
                                    <LatencySparkline values={recentLatencies} />
                                </div>
                                <div className="flex flex-wrap gap-2">
                                    {recentRuns.map(run => (
                                        <button
                                            key={run.at}
                                            onClick={() => handleRestoreRun(run)}
                                            className="flex items-center gap-2 text-[11px] px-2 py-1 rounded border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-colors"
                                            title="Restore inputs and predictions from this run"
                                        >
                                            <span className="font-mono">{formatTime(run.at)}</span>
                                            <span className="text-gray-400">·</span>
                                            <span>
                                                {run.rows} row{run.rows === 1 ? '' : 's'}
                                            </span>
                                            <span className="text-gray-400">·</span>
                                            <span className="tabular-nums">{run.latencyMs} ms</span>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
