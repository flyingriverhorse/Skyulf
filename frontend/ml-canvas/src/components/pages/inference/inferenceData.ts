export const DEFAULT_INPUT = '[\n  {\n    "feature1": 0.5,\n    "feature2": 1.2\n  }\n]';
export const MAX_RECENT_RUNS = 5;
export const SAMPLE_OPTIONS: ReadonlyArray<number> = [1, 5, 10, 25, 100];
export const HISTOGRAM_BINS = 10;
export const LARGE_BATCH_THRESHOLD = 500;
// EXP-007: a hung request should surface as a distinct, actionable timeout
// rather than spinning forever with no explicit recovery path.
export const PREDICT_TIMEOUT_MS = 30_000;
// How long a settled run is kept in the durable, reload-surviving history
// before it's treated as expired — bounds how long prediction inputs/results
// linger in browser storage for privacy.
export const HISTORY_TTL_MS = 24 * 60 * 60 * 1000;

export const LS_INPUT = 'inferencePage:lastInput';
export const LS_SAMPLE_SIZE = 'inferencePage:sampleSize';
export const LS_VIEW = 'inferencePage:resultsView';
export const LS_RUN_HISTORY = 'inferencePage:runHistory';
export const LS_PENDING_RUN = 'inferencePage:pendingRun';

/** Live status of the JSON typed into the textarea. */
export interface InputStatus {
    valid: boolean;
    rows: number;
    message: string;
    /** 1-based line number when JSON parse fails, if extractable. */
    line?: number;
    column?: number;
}

/** Per-row breakdown of which schema fields a specific row omits. */
export interface RowSchemaIssue {
    rowIndex: number; // 0-based index into the parsed input array
    missing: string[];
}

/** Schema-vs-input drift summary surfaced under the editor. */
export interface SchemaCheck {
    missing: string[]; // schema fields that no row provides at all
    extra: string[]; // row fields that the schema does not declare
    rows: number;
    /** Rows that are individually missing at least one schema field — even
     * when every field is *somewhere* present across the batch, a single
     * row silently sent with a hole in it is still the exact request the
     * backend rejects (or, worse, crashes on). Named per row/field so the
     * user can see and fix the actual offending rows, not just an
     * aggregate count. */
    rowIssues: RowSchemaIssue[];
}

/** Which threshold source, if any, was in effect for a given run. */
export type RunThresholdContext = 'none' | 'saved-enabled' | 'override';

/** How a settled prediction run ended up. */
export type RunOutcome = 'success' | 'failure' | 'cancelled';

/**
 * A single named inference run's full provenance: which model/version,
 * which input, when, under what threshold context, and — once settled —
 * what it produced or why it didn't. This is the durable unit EXP-007
 * requires in place of a bare `predictions`/`error` pair, so a user (or a
 * reviewer later) can always answer "which model, which input, which
 * thresholds, when" for anything shown on this page.
 */
export interface RunRecord {
    runId: string;
    /** Short, stable display name, e.g. "Run #3". */
    label: string;
    status: RunOutcome;
    /** Epoch ms when the run settled (succeeded, failed, or was cancelled). */
    at: number;
    rows: number;
    /** Client-observed round-trip latency; null when the run never got a response. */
    latencyMs: number | null;
    jobId: string;
    modelType: string;
    /** Server-reported model/version identifier, when the run succeeded. */
    modelVersion: string | null;
    thresholdContext: RunThresholdContext;
    /** Exact JSON payload submitted — kept so "retry" always resends the
     * request that actually failed, even if the textarea has since changed. */
    input: string;
    overrideThresholdsUsed: Record<string, number> | null;
    predictions: unknown[] | null;
    /** Safe, human-readable cause — never the raw transport/error object. */
    errorMessage: string | null;
}

/** A run that is currently in flight, named so it can't be double-submitted. */
export interface PendingRun {
    runId: string;
    label: string;
    submittedAt: number;
    /** Set when this pending run is a retry of an earlier run. */
    retryOf: string | null;
}

/** Best-effort read of the durable run history from localStorage, dropping
 * anything past `HISTORY_TTL_MS` so retention/expiry stays explicit rather
 * than silently accumulating prediction inputs/results forever. */
export const loadRunHistory = (): RunRecord[] => {
    try {
        const raw = localStorage.getItem(LS_RUN_HISTORY);
        if (!raw) return [];
        const parsed: unknown = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        const now = Date.now();
        return (parsed as RunRecord[])
            .filter(entry => entry && typeof entry.at === 'number' && now - entry.at < HISTORY_TTL_MS)
            .slice(0, MAX_RECENT_RUNS);
    } catch {
        return [];
    }
};

/** Persist the run history, best-effort (storage may be disabled/full). */
export const persistRunHistory = (entries: RunRecord[]): void => {
    try {
        localStorage.setItem(LS_RUN_HISTORY, JSON.stringify(entries));
    } catch {
        /* storage may be disabled in private mode — ignore */
    }
};

/** True when `error` is an axios cancellation (user cancel or our own timeout-abort). */
export const isAbortError = (error: unknown): boolean =>
    Boolean(
        error &&
        typeof error === 'object' &&
        ('code' in error) &&
        (error as { code?: unknown }).code === 'ERR_CANCELED',
    );

/** Try to extract a (line, column) tuple from a JSON parse error message. */
export const extractJsonErrorPosition = (
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
export const analyseInput = (raw: string): InputStatus => {
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
export const checkSchema = (raw: string, schema: { name: string; type: string }[]): SchemaCheck | null => {
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

    // Per-row detail: a field can be "not missing overall" (some other row
    // provides it) while this specific row still omits it — that row would
    // still be sent with a hole in it, so surface it individually.
    const rowIssues: RowSchemaIssue[] = [];
    parsed.forEach((row, rowIndex) => {
        const keys =
            row && typeof row === 'object' ? new Set(Object.keys(row as Record<string, unknown>)) : new Set<string>();
        const rowMissing = [...schemaNames].filter(n => !keys.has(n));
        if (rowMissing.length > 0) rowIssues.push({ rowIndex, missing: rowMissing });
    });

    return { missing, extra, rows: parsed.length, rowIssues };
};

/**
 * Numeric-looking cell? Accepts ints, decimals, scientific notation, signed.
 * The earlier `raw === String(num)` round-trip rejected `"1.0"` / `"1e5"` and
 * left them as strings, which then crashed the model with
 * `unsupported operand type(s) for -: 'str' and 'float'`.
 */
export const NUMERIC_RE = /^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

/**
 * Tiny CSV-to-objects parser: comma-separated, first row is headers, no
 * quoted-field handling. Numeric-looking cells are coerced to numbers.
 */
export const parseCsv = (text: string): Record<string, unknown>[] => {
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
export const toNumericArray = (preds: unknown[]): number[] =>
    preds
        .map(p => (typeof p === 'number' ? p : Number(p)))
        .filter(n => Number.isFinite(n));

/** Human label for a run's threshold context — shown in run provenance. */
export const describeThresholdContext = (ctx: RunThresholdContext): string => {
    switch (ctx) {
        case 'override':
            return 'ad-hoc override thresholds';
        case 'saved-enabled':
            return 'saved tuned thresholds';
        default:
            return 'default thresholds';
    }
};

/** Format an epoch ms as HH:MM:SS for the recent-runs strip. */
export const formatTime = (ms: number): string =>
    new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

/** Best-effort short string rendering for a single prediction value. */
export const renderPrediction = (pred: unknown): string =>
    typeof pred === 'object' && pred !== null ? JSON.stringify(pred) : String(pred);

/** Detect classification probability shape: {classLabel: number, ...}. */
export const asProbabilityMap = (pred: unknown): Record<string, number> | null => {
    if (!pred || typeof pred !== 'object' || Array.isArray(pred)) return null;
    const obj = pred as Record<string, unknown>;
    const entries = Object.entries(obj);
    if (entries.length === 0) return null;
    const numeric: Record<string, number> = {};
    for (const [k, v] of entries) {
        if (!isFiniteNumber(v)) return null;
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
export const projectSampleRow = (
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
export const rowsToCsv = (rows: Record<string, unknown>[]): string => {
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

const isFiniteNumber = (value: unknown): value is number =>
    typeof value === 'number' && Number.isFinite(value);
