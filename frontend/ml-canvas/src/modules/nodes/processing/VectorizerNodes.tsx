import React, { useState } from 'react';
import { NodeDefinition, NodeSettingsProps } from '../../../core/types/nodes';
import { Hash, FileText, Binary, Search, Info } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../core/hooks/useUpstreamDroppedColumns';
import { clickableProps } from '../../../core/utils/a11y';

// ── Shared config types ───────────────────────────────────────────────────────

interface BaseVectorizerConfig {
  columns: string[];
  drop_original: boolean;
  lowercase: boolean;
  stop_words: 'english' | null;
}

interface CountVectorizerConfig extends BaseVectorizerConfig {
  max_features: number | null;
  min_df: number;
  max_df: number;
  ngram_range: [number, number];
  binary?: boolean;
}

interface TfidfVectorizerConfig extends CountVectorizerConfig {
  sublinear_tf: boolean;
}

interface HashingVectorizerConfig extends BaseVectorizerConfig {
  n_features: number;
  norm: 'l1' | 'l2' | 'none';
  alternate_sign: boolean;
}

type AnyVectorizerConfig =
  | CountVectorizerConfig
  | TfidfVectorizerConfig
  | HashingVectorizerConfig;

// ── Reusable text-column selector ─────────────────────────────────────────────

export const ColumnSelector: React.FC<{
  columns: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
}> = ({ columns, selected, onChange }) => {
  const [search, setSearch] = useState('');
  const filtered = columns.filter((c) => c.toLowerCase().includes(search.toLowerCase()));

  const toggle = (col: string) => {
    if (selected.includes(col)) {
      onChange(selected.filter((c) => c !== col));
    } else {
      onChange([...selected, col]);
    }
  };

  return (
    <div className="border rounded bg-background overflow-hidden flex flex-col max-h-40">
      <div className="flex items-center px-2 py-1.5 border-b bg-muted/20">
        <Search size={12} className="text-muted-foreground mr-1.5" />
        <input
          className="flex-1 bg-transparent text-xs outline-none placeholder:text-muted-foreground/70"
          placeholder="Search columns..."
          value={search}
          onChange={(e) => { setSearch(e.target.value); }}
        />
      </div>
      <div className="overflow-y-auto p-1 space-y-0.5">
        {filtered.length > 0 ? (
          filtered.map((col) => {
            const isSelected = selected.includes(col);
            return (
              <div
                key={col}
                {...clickableProps(() => { toggle(col); })}
                className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs cursor-pointer transition-colors ${
                  isSelected
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 font-medium'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
                }`}
              >
                <div
                  className={`w-3 h-3 rounded border flex items-center justify-center ${
                    isSelected ? 'border-blue-500 bg-blue-500 text-white' : 'border-gray-400 dark:border-gray-600'
                  }`}
                >
                  {isSelected && <div className="w-1.5 h-1.5 bg-white rounded-full" />}
                </div>
                <span className="truncate">{col}</span>
              </div>
            );
          })
        ) : (
          <div className="p-2 text-xs text-gray-500 text-center italic">No columns found</div>
        )}
      </div>
    </div>
  );
};

// ── Small labelled field helpers ──────────────────────────────────────────────

export const NumberField: React.FC<{
  label: string;
  value: number | null;
  placeholder?: string;
  min?: number;
  step?: number;
  onChange: (v: number | null) => void;
  hint?: string;
}> = ({ label, value, placeholder, min, step, onChange, hint }) => (
  <div>
    <label className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1 block">{label}</label>
    <input
      type="number"
      className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5"
      value={value ?? ''}
      placeholder={placeholder}
      min={min}
      step={step}
      onChange={(e) => {
        const raw = e.target.value;
        onChange(raw === '' ? null : Number(raw));
      }}
    />
    {hint && <p className="text-[11px] text-gray-500 mt-0.5">{hint}</p>}
  </div>
);

export const Checkbox: React.FC<{
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
}> = ({ label, checked, onChange, hint }) => (
  <label className="flex items-start gap-2 text-xs text-gray-700 dark:text-gray-300 cursor-pointer">
    <input
      type="checkbox"
      className="mt-0.5"
      checked={checked}
      onChange={(e) => { onChange(e.target.checked); }}
    />
    <span>
      {label}
      {hint && <span className="block text-[11px] text-gray-500">{hint}</span>}
    </span>
  </label>
);

// ── Shared settings component (variant-aware) ─────────────────────────────────

type Variant = 'count' | 'tfidf' | 'hashing';

const INFO_TEXT: Record<Variant, string> = {
  count:
    'Converts text into token-count columns (bag-of-words). Fits a vocabulary on the training data, then counts how often each token appears per row.',
  tfidf:
    'Converts text into TF-IDF weighted columns. Down-weights common tokens and emphasises distinctive ones — the most common starting point for text classification.',
  hashing:
    'Stateless hashing vectorizer. Maps tokens to a fixed number of columns via a hash function — no vocabulary stored, so it scales to huge corpora and unseen tokens.',
};

const VectorizerSettings: React.FC<NodeSettingsProps<AnyVectorizerConfig> & { variant: Variant }> = ({
  config,
  onChange,
  nodeId,
  variant,
  isExpanded,
}) => {
  // Info box starts open only in the roomy expanded panel; in the narrow
  // w-80 panel it starts collapsed so the form stays short.
  const [showInfo, setShowInfo] = useState(isExpanded ?? false);

  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find((d: Record<string, unknown>) => d.datasetId)?.datasetId as
    | string
    | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);

  const textColumns = schema
    ? Object.values(schema.columns)
        .filter((c) => {
          const dtype = String(c.dtype).toLowerCase();
          return (
            dtype.includes('object') ||
            dtype.includes('string') ||
            dtype.includes('category') ||
            dtype.includes('text')
          );
        })
        .filter((c) => !droppedUpstream.has(c.name))
        .map((c) => c.name)
    : [];

  // Narrowed accessors with safe fallbacks (configs share the base shape).
  const countCfg = config as CountVectorizerConfig;
  const tfidfCfg = config as TfidfVectorizerConfig;
  const hashCfg = config as HashingVectorizerConfig;

  const patch = (changes: Partial<AnyVectorizerConfig>) =>
    onChange({ ...config, ...changes } as AnyVectorizerConfig);

  return (
    <div className="flex flex-col h-full w-full bg-white dark:bg-gray-900 overflow-y-auto">
      <div
        className={`flex-1 min-h-0 p-4 grid grid-cols-1 gap-4 ${
          isExpanded ? 'md:grid-cols-2' : ''
        }`}
      >
        {/* Left: columns */}
        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
              Text Columns ({config.columns.length})
            </label>
            <ColumnSelector
              columns={textColumns}
              selected={config.columns}
              onChange={(cols) => patch({ columns: cols })}
            />
            <p className="text-xs text-gray-500 mt-1">
              Only text/categorical columns are shown. Multiple columns are joined with a space.
            </p>
          </div>

          <Checkbox
            label="Drop original text column(s)"
            checked={config.drop_original}
            onChange={(v) => patch({ drop_original: v })}
            hint="Remove the source text after vectorizing."
          />
        </div>

        {/* Right: parameters */}
        <div
          className={`space-y-3 ${
            isExpanded ? 'md:border-l md:border-gray-100 md:dark:border-gray-700 md:pl-4' : ''
          }`}
        >
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-md border border-blue-100 dark:border-blue-800 overflow-hidden">
            <button
              onClick={() => { setShowInfo(!showInfo); }}
              className="w-full flex items-center gap-2 p-3 text-left hover:bg-blue-100/50 dark:hover:bg-blue-900/30 transition-colors"
            >
              <Info className="text-blue-600 dark:text-blue-400 shrink-0" size={16} />
              <span className="text-xs font-semibold text-blue-800 dark:text-blue-200 flex-1">
                About this vectorizer
              </span>
            </button>
            {showInfo && (
              <div className="px-3 pb-3 text-xs text-blue-800 dark:text-blue-200 pl-9">
                <p>{INFO_TEXT[variant]}</p>
              </div>
            )}
          </div>

          <Checkbox
            label="Lowercase"
            checked={config.lowercase ?? true}
            onChange={(v) => patch({ lowercase: v })}
            hint="Lowercase text before tokenizing."
          />
          <Checkbox
            label="Remove English stop words"
            checked={config.stop_words === 'english'}
            onChange={(v) => patch({ stop_words: v ? 'english' : null })}
            hint="Drop common words like 'the', 'and' (word analyzer only)."
          />

          {variant === 'hashing' ? (
            <>
              <NumberField
                label="Number of features (hash buckets)"
                value={hashCfg.n_features ?? 1024}
                min={2}
                step={1}
                onChange={(v) => patch({ n_features: v ?? 1024 } as Partial<HashingVectorizerConfig>)}
                hint="Higher = fewer collisions, more columns. Powers of 2 recommended."
              />
              <div>
                <label className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1 block">
                  Normalization
                  <select
                    className="mt-1 w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5"
                    value={hashCfg.norm ?? 'l2'}
                    onChange={(e) =>
                      patch({ norm: e.target.value as HashingVectorizerConfig['norm'] } as Partial<HashingVectorizerConfig>)
                    }
                  >
                    <option value="l2">L2</option>
                    <option value="l1">L1</option>
                    <option value="none">None</option>
                  </select>
                </label>
              </div>
              <Checkbox
                label="Alternate sign"
                checked={hashCfg.alternate_sign ?? true}
                onChange={(v) =>
                  patch({ alternate_sign: v } as Partial<HashingVectorizerConfig>)
                }
                hint="Adds +/- signs to approximately preserve inner products."
              />
            </>
          ) : (
            <>
              <NumberField
                label="Max features"
                value={countCfg.max_features ?? null}
                placeholder="unlimited"
                min={1}
                step={1}
                onChange={(v) => patch({ max_features: v } as Partial<CountVectorizerConfig>)}
                hint="Keep only the top-N most frequent tokens. Empty = no limit."
              />
              <div className="grid grid-cols-2 gap-2">
                <NumberField
                  label="Min document freq"
                  value={countCfg.min_df ?? 1}
                  min={1}
                  step={1}
                  onChange={(v) => patch({ min_df: v ?? 1 } as Partial<CountVectorizerConfig>)}
                  hint="Ignore rarer tokens."
                />
                <NumberField
                  label="Max document freq"
                  value={countCfg.max_df ?? 1.0}
                  min={0}
                  step={0.05}
                  onChange={(v) => patch({ max_df: v ?? 1.0 } as Partial<CountVectorizerConfig>)}
                  hint="Drop very common tokens (0–1 = ratio)."
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <NumberField
                  label="N-gram min"
                  value={countCfg.ngram_range?.[0] ?? 1}
                  min={1}
                  step={1}
                  onChange={(v) =>
                    patch({
                      ngram_range: [v ?? 1, countCfg.ngram_range?.[1] ?? 1],
                    } as Partial<CountVectorizerConfig>)
                  }
                />
                <NumberField
                  label="N-gram max"
                  value={countCfg.ngram_range?.[1] ?? 1}
                  min={1}
                  step={1}
                  onChange={(v) =>
                    patch({
                      ngram_range: [countCfg.ngram_range?.[0] ?? 1, v ?? 1],
                    } as Partial<CountVectorizerConfig>)
                  }
                />
              </div>
              {variant === 'count' && (
                <Checkbox
                  label="Binary counts"
                  checked={countCfg.binary ?? false}
                  onChange={(v) => patch({ binary: v } as Partial<CountVectorizerConfig>)}
                  hint="Use 1/0 presence instead of raw token counts."
                />
              )}
              {variant === 'tfidf' && (
                <Checkbox
                  label="Sublinear TF scaling"
                  checked={tfidfCfg.sublinear_tf ?? false}
                  onChange={(v) =>
                    patch({ sublinear_tf: v } as Partial<TfidfVectorizerConfig>)
                  }
                  hint="Replace term frequency tf with 1 + log(tf)."
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// ── Body previews ─────────────────────────────────────────────────────────────

const bagPreview = (config: CountVectorizerConfig): string | null => {
  const cols = config.columns?.length ?? 0;
  if (cols === 0) return null;
  const cap = config.max_features ? `≤${config.max_features}` : 'all';
  return `${cols} ${cols === 1 ? 'col' : 'cols'} · ${cap} feats`;
};

const baseValidate = (config: { columns: string[] }) =>
  config.columns.length === 0
    ? { isValid: false, message: 'Select at least one text column.' }
    : { isValid: true };

// ── Node definitions ──────────────────────────────────────────────────────────

export const CountVectorizerNode: NodeDefinition<CountVectorizerConfig> = {
  type: 'count_vectorizer',
  label: 'Count Vectorizer',
  category: 'Preprocessing',
  description: 'Convert text to token-count columns (bag-of-words).',
  icon: Hash,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Vectorized', type: 'dataset' }],
  settings: (props) => (
    <VectorizerSettings {...(props as NodeSettingsProps<AnyVectorizerConfig>)} variant="count" />
  ),
  bodyPreview: bagPreview,
  validate: baseValidate,
  getDefaultConfig: () => ({
    columns: [],
    max_features: null,
    min_df: 1,
    max_df: 1.0,
    ngram_range: [1, 1],
    lowercase: true,
    stop_words: null,
    binary: false,
    drop_original: false,
  }),
};

export const TfidfVectorizerNode: NodeDefinition<TfidfVectorizerConfig> = {
  type: 'tfidf_vectorizer',
  label: 'TF-IDF Vectorizer',
  category: 'Preprocessing',
  description: 'Convert text to TF-IDF weighted feature columns.',
  icon: FileText,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Vectorized', type: 'dataset' }],
  settings: (props) => (
    <VectorizerSettings {...(props as NodeSettingsProps<AnyVectorizerConfig>)} variant="tfidf" />
  ),
  bodyPreview: bagPreview,
  validate: baseValidate,
  getDefaultConfig: () => ({
    columns: [],
    max_features: null,
    min_df: 1,
    max_df: 1.0,
    ngram_range: [1, 1],
    sublinear_tf: false,
    lowercase: true,
    stop_words: null,
    drop_original: false,
  }),
};

export const HashingVectorizerNode: NodeDefinition<HashingVectorizerConfig> = {
  type: 'hashing_vectorizer',
  label: 'Hashing Vectorizer',
  category: 'Preprocessing',
  description: 'Stateless hash-trick text vectorizer (fixed feature size).',
  icon: Binary,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Vectorized', type: 'dataset' }],
  settings: (props) => (
    <VectorizerSettings {...(props as NodeSettingsProps<AnyVectorizerConfig>)} variant="hashing" />
  ),
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    if (cols === 0) return null;
    return `${cols} ${cols === 1 ? 'col' : 'cols'} · ${config.n_features ?? 1024} buckets`;
  },
  validate: baseValidate,
  getDefaultConfig: () => ({
    columns: [],
    n_features: 1024,
    norm: 'l2',
    alternate_sign: true,
    lowercase: true,
    stop_words: null,
    drop_original: false,
  }),
};
