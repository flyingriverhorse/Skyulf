import React, { useState } from 'react';
import { NodeDefinition, NodeSettingsProps } from '../../../core/types/nodes';
import { Scissors, Brain, Info, AlertTriangle } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../core/hooks/useUpstreamDroppedColumns';
import { ColumnSelector, NumberField, Checkbox } from './VectorizerNodes';

// ── Config types ──────────────────────────────────────────────────────────────

interface TokenizerConfig {
  columns: string[];
  analyzer: 'word' | 'char' | 'char_wb';
  lowercase: boolean;
  stop_words: 'english' | null;
  ngram_range: [number, number];
  add_token_count: boolean;
  drop_original: boolean;
}

interface SentenceEmbedderConfig {
  columns: string[];
  model_name: string;
  normalize: boolean;
  drop_original: boolean;
}

// ── Shared: text-column discovery from upstream schema ────────────────────────

const useTextColumns = (nodeId?: string): string[] => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find((d: Record<string, unknown>) => d.datasetId)?.datasetId as
    | string
    | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);

  return schema
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
};

const InfoBox: React.FC<{ text: string; defaultOpen?: boolean }> = ({ text, defaultOpen = true }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-md border border-blue-100 dark:border-blue-800 overflow-hidden">
      <button
        onClick={() => { setOpen(!open); }}
        className="w-full flex items-center gap-2 p-3 text-left hover:bg-blue-100/50 dark:hover:bg-blue-900/30 transition-colors"
      >
        <Info className="text-blue-600 dark:text-blue-400 shrink-0" size={16} />
        <span className="text-xs font-semibold text-blue-800 dark:text-blue-200 flex-1">
          About this node
        </span>
      </button>
      {open && (
        <div className="px-3 pb-3 text-xs text-blue-800 dark:text-blue-200 pl-9">
          <p>{text}</p>
        </div>
      )}
    </div>
  );
};

const ColumnsPanel: React.FC<{
  columns: string[];
  selected: string[];
  dropOriginal: boolean;
  onColumns: (cols: string[]) => void;
  onDrop: (v: boolean) => void;
}> = ({ columns, selected, dropOriginal, onColumns, onDrop }) => (
  <div className="space-y-4">
    <div>
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
        Text Columns ({selected.length})
      </label>
      <ColumnSelector columns={columns} selected={selected} onChange={onColumns} />
      <p className="text-xs text-gray-500 mt-1">
        Only text/categorical columns are shown. Multiple columns are joined with a space.
      </p>
    </div>
    <Checkbox
      label="Drop original text column(s)"
      checked={dropOriginal}
      onChange={onDrop}
      hint="Remove the source text after processing."
    />
  </div>
);

// ── Tokenizer settings ────────────────────────────────────────────────────────

const TokenizerSettings: React.FC<NodeSettingsProps<TokenizerConfig>> = ({
  config,
  onChange,
  nodeId,
  isExpanded,
}) => {
  const textColumns = useTextColumns(nodeId);
  const patch = (changes: Partial<TokenizerConfig>) => onChange({ ...config, ...changes });

  return (
    <div className="flex flex-col h-full w-full bg-white dark:bg-gray-900 overflow-y-auto">
      <div
        className={`flex-1 min-h-0 p-4 grid grid-cols-1 gap-4 ${
          isExpanded ? 'md:grid-cols-2' : ''
        }`}
      >
        <ColumnsPanel
          columns={textColumns}
          selected={config.columns}
          dropOriginal={config.drop_original}
          onColumns={(cols) => patch({ columns: cols })}
          onDrop={(v) => patch({ drop_original: v })}
        />

        <div
          className={`space-y-3 ${
            isExpanded ? 'md:border-l md:border-gray-100 md:dark:border-gray-700 md:pl-4' : ''
          }`}
        >
          <InfoBox
            defaultOpen={isExpanded ?? false}
            text="Inspection / intermediate tool. Splits text into tokens and outputs a space-joined token string column per source column (optionally a token-count column). Stateless — no vocabulary is fitted."
          />

          <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-900/20 p-3">
            <AlertTriangle className="text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" size={16} />
            <p className="text-xs text-amber-800 dark:text-amber-200">
              Don&apos;t feed this into a vectorizer. The Count / TF-IDF / Hashing vectorizers already
              tokenize internally — chaining them leaves a stray text column that breaks training. Use a
              vectorizer directly; reach for this node only to inspect tokens or add a token-count feature.
            </p>
          </div>

          <div>
            <label className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1 block">
              Analyzer
              <select
                className="mt-1 w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5"
                value={config.analyzer}
                onChange={(e) => patch({ analyzer: e.target.value as TokenizerConfig['analyzer'] })}
              >
                <option value="word">Word</option>
                <option value="char">Character</option>
                <option value="char_wb">Character (within word boundaries)</option>
              </select>
            </label>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <NumberField
              label="N-gram min"
              value={config.ngram_range?.[0] ?? 1}
              min={1}
              step={1}
              onChange={(v) => patch({ ngram_range: [v ?? 1, config.ngram_range?.[1] ?? 1] })}
            />
            <NumberField
              label="N-gram max"
              value={config.ngram_range?.[1] ?? 1}
              min={1}
              step={1}
              onChange={(v) => patch({ ngram_range: [config.ngram_range?.[0] ?? 1, v ?? 1] })}
            />
          </div>

          <Checkbox
            label="Lowercase"
            checked={config.lowercase}
            onChange={(v) => patch({ lowercase: v })}
          />
          <Checkbox
            label="Remove English stop words"
            checked={config.stop_words === 'english'}
            onChange={(v) => patch({ stop_words: v ? 'english' : null })}
            hint="Drop common words like 'the', 'and' (word analyzer only)."
          />
          <Checkbox
            label="Add token-count column"
            checked={config.add_token_count}
            onChange={(v) => patch({ add_token_count: v })}
            hint="Emit a {col}__token_count numeric column."
          />
        </div>
      </div>
    </div>
  );
};

// ── SentenceEmbedder settings ─────────────────────────────────────────────────

const EMBED_MODELS: { value: string; label: string }[] = [
  { value: 'all-MiniLM-L6-v2', label: 'all-MiniLM-L6-v2 (384d, fast)' },
  { value: 'all-mpnet-base-v2', label: 'all-mpnet-base-v2 (768d, accurate)' },
  { value: 'paraphrase-multilingual-MiniLM-L12-v2', label: 'multilingual-MiniLM (384d)' },
];

const SentenceEmbedderSettings: React.FC<NodeSettingsProps<SentenceEmbedderConfig>> = ({
  config,
  onChange,
  nodeId,
  isExpanded,
}) => {
  const textColumns = useTextColumns(nodeId);
  const patch = (changes: Partial<SentenceEmbedderConfig>) => onChange({ ...config, ...changes });

  return (
    <div className="flex flex-col h-full w-full bg-white dark:bg-gray-900 overflow-y-auto">
      <div
        className={`flex-1 min-h-0 p-4 grid grid-cols-1 gap-4 ${
          isExpanded ? 'md:grid-cols-2' : ''
        }`}
      >
        <ColumnsPanel
          columns={textColumns}
          selected={config.columns}
          dropOriginal={config.drop_original}
          onColumns={(cols) => patch({ columns: cols })}
          onDrop={(v) => patch({ drop_original: v })}
        />

        <div
          className={`space-y-3 ${
            isExpanded ? 'md:border-l md:border-gray-100 md:dark:border-gray-700 md:pl-4' : ''
          }`}
        >
          <InfoBox
            defaultOpen={isExpanded ?? false}
            text="Encodes text into dense semantic embeddings using a sentence-transformers model. Requires the optional 'sentence-transformers' package on the backend (install skyulf[nlp])."
          />

          <div>
            <label className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1 block">
              Model
              <select
                className="mt-1 w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5"
                value={config.model_name}
                onChange={(e) => patch({ model_name: e.target.value })}
              >
                {EMBED_MODELS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Downloaded on first use and cached on the backend.
            </p>
          </div>

          <Checkbox
            label="Normalize embeddings"
            checked={config.normalize}
            onChange={(v) => patch({ normalize: v })}
            hint="L2-normalize vectors (recommended for cosine similarity)."
          />
        </div>
      </div>
    </div>
  );
};

// ── Node definitions ──────────────────────────────────────────────────────────

export const TokenizerNode: NodeDefinition<TokenizerConfig> = {
  type: 'tokenizer',
  label: 'Tokenizer',
  category: 'Preprocessing',
  description: 'Split text into tokens (word / char), optional token counts.',
  icon: Scissors,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Tokenized', type: 'dataset' }],
  settings: TokenizerSettings,
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    if (cols === 0) return null;
    return `${cols} ${cols === 1 ? 'col' : 'cols'} · ${config.analyzer}`;
  },
  validate: (config) =>
    config.columns.length === 0
      ? { isValid: false, message: 'Select at least one text column.' }
      : { isValid: true },
  getDefaultConfig: () => ({
    columns: [],
    analyzer: 'word',
    lowercase: true,
    stop_words: null,
    ngram_range: [1, 1],
    add_token_count: false,
    drop_original: false,
  }),
};

export const SentenceEmbedderNode: NodeDefinition<SentenceEmbedderConfig> = {
  type: 'sentence_embedder',
  label: 'Sentence Embedder',
  category: 'Preprocessing',
  description: 'Dense semantic embeddings via sentence-transformers (optional dep).',
  icon: Brain,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Embeddings', type: 'dataset' }],
  settings: SentenceEmbedderSettings,
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    if (cols === 0) return null;
    return `${cols} ${cols === 1 ? 'col' : 'cols'} · ${config.model_name}`;
  },
  validate: (config) =>
    config.columns.length === 0
      ? { isValid: false, message: 'Select at least one text column.' }
      : { isValid: true },
  getDefaultConfig: () => ({
    columns: [],
    model_name: 'all-MiniLM-L6-v2',
    normalize: true,
    drop_original: false,
  }),
};
