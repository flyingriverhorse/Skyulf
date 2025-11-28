import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import type { ModelHyperparameterField } from '../../../../api';
import { formatRelativeTime } from '../../utils/formatters';

export type HyperparamPreset = {
  id: string;
  label: string;
  source: 'best-api' | 'tuning-job';
  modelType: string;
  params: Record<string, any>;
  jobId?: string | null;
  nodeId?: string | null;
  runNumber?: number | null;
  score?: number | null;
  scoring?: string | null;
  finishedAt?: string | null;
  searchStrategy?: string | null;
  nIterations?: number | null;
  targetColumn?: string | null;
  pipelineId?: string | null;
  datasetSourceId?: string | null;
  description?: string | null;
};

type BestHyperparamsModalProps = {
  presets: HyperparamPreset[];
  isOpen: boolean;
  onClose: () => void;
  onApply: (preset: HyperparamPreset) => void;
  activePresetId?: string | null;
  fieldMetadata?: Record<string, ModelHyperparameterField>;
};

const formatPresetScore = (score?: number | null): string => {
  if (typeof score !== 'number' || !Number.isFinite(score)) {
    return '—';
  }
  if (Math.abs(score) >= 1000) {
    return score.toFixed(0);
  }
  if (Math.abs(score) >= 100) {
    return score.toFixed(1);
  }
  if (Math.abs(score) >= 10) {
    return score.toFixed(2);
  }
  return score.toFixed(3);
};

export const BestHyperparamsModal: React.FC<BestHyperparamsModalProps> = ({
  presets,
  isOpen,
  onClose,
  onApply,
  activePresetId,
  fieldMetadata,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setSelectedId(null);
      return;
    }
    if (selectedId && presets.some((preset) => preset.id === selectedId)) {
      return;
    }
    setSelectedId(presets[0]?.id ?? null);
  }, [isOpen, presets, selectedId]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  const selectedPreset = useMemo(() => {
    if (!selectedId) {
      return presets[0] ?? null;
    }
    return presets.find((preset) => preset.id === selectedId) ?? presets[0] ?? null;
  }, [presets, selectedId]);

  const currentIsApplied = useMemo(() => {
    if (!activePresetId || !selectedPreset) {
      return false;
    }
    return activePresetId === selectedPreset.id;
  }, [activePresetId, selectedPreset]);

  const content = useMemo(() => {
    if (!isOpen || !presets.length) {
      return null;
    }

    const logisticSolverRaw =
      selectedPreset?.modelType?.toLowerCase() === 'logistic_regression'
        ? selectedPreset?.params?.solver
        : null;
    const logisticSolver =
      typeof logisticSolverRaw === 'string' ? logisticSolverRaw.trim().toLowerCase() : null;
    const scalingNotice =
      logisticSolver && (logisticSolver === 'sag' || logisticSolver === 'saga')
        ? 'SAG/SAGA solvers require scaled features for stability.'
        : null;

    const paramEntries = selectedPreset
      ? Object.entries(selectedPreset.params ?? {})
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([key, value]) => {
            const metadata = fieldMetadata?.[key];

            const findOption = (rawValue: any) => {
              if (!metadata?.options?.length) {
                return null;
              }
              return (
                metadata.options.find((option) => {
                  const optionValue = option.value;
                  if (rawValue === null || rawValue === undefined) {
                    if (optionValue === null) {
                      return true;
                    }
                    if (
                      typeof optionValue === 'string' && optionValue.trim().toLowerCase() === 'none'
                    ) {
                      return true;
                    }
                    return false;
                  }
                  if (typeof rawValue === 'string' && typeof optionValue === 'string') {
                    if (optionValue === rawValue) {
                      return true;
                    }
                    return optionValue.trim().toLowerCase() === rawValue.trim().toLowerCase();
                  }
                  return String(optionValue) === String(rawValue);
                }) ?? null
              );
            };

            const formatPrimitive = (entry: any): string => {
              if (entry === null) {
                return metadata?.nullable ? 'None' : '—';
              }
              if (entry === undefined) {
                return '—';
              }
              if (typeof entry === 'boolean') {
                return entry ? 'true' : 'false';
              }
              if (typeof entry === 'number') {
                return Number.isFinite(entry) ? entry.toString() : String(entry);
              }
              if (entry && typeof entry === 'object') {
                try {
                  return JSON.stringify(entry);
                } catch (error) {
                  return '[object]';
                }
              }
              return String(entry ?? '');
            };

            const formatOption = (option: { label?: string; value?: any } | null) => {
              if (!option) {
                return null;
              }
              const optionValue = option.value;
              const label = option.label?.trim();
              const valueText = formatPrimitive(optionValue);
              if (label && valueText && label !== valueText) {
                return `${label} (${valueText})`;
              }
              return label || valueText;
            };

            const fallbackToDefault = (): { text: string; note: string | null } => {
              if (metadata && metadata.default !== undefined) {
                const formatted = resolveValue(metadata.default, false);
                if (formatted.text) {
                  return {
                    text: formatted.text,
                    note: 'Using estimator default',
                  };
                }
              }
              return {
                text: '—',
                note: 'Using estimator default',
              };
            };

            const resolveValue = (entry: any, allowFallback = true): { text: string; note: string | null } => {
              const optionMatch = findOption(entry);
              if (optionMatch) {
                return {
                  text: formatOption(optionMatch) ?? '—',
                  note: null,
                };
              }

              if (Array.isArray(entry)) {
                if (entry.length === 0) {
                  return allowFallback ? fallbackToDefault() : { text: '—', note: null };
                }
                const formattedItems = entry.map((item) => resolveValue(item, false));
                return {
                  text: formattedItems.map((item) => item.text).filter(Boolean).join(', '),
                  note: null,
                };
              }

              if (entry === undefined || (typeof entry === 'string' && entry.trim() === '')) {
                return allowFallback ? fallbackToDefault() : { text: '—', note: null };
              }

              if (entry === null) {
                if (metadata?.nullable) {
                  return { text: 'None', note: null };
                }
                return allowFallback ? fallbackToDefault() : { text: '—', note: null };
              }

              return {
                text: formatPrimitive(entry),
                note: null,
              };
            };

            const resolved = resolveValue(value);

            return {
              key,
              label: metadata?.label ?? key,
              description: metadata?.description ?? null,
              value: resolved.text,
              note: resolved.note,
            };
          })
      : [];

    const relativeFinished = selectedPreset?.finishedAt
      ? formatRelativeTime(selectedPreset.finishedAt)
      : null;

    return (
      <div className="canvas-modal best-param-modal" role="dialog" aria-modal="true">
        <div className="canvas-modal__backdrop" onClick={onClose} />
        <div className="canvas-modal__panel best-param-modal__panel" role="document">
          <header className="best-param-modal__header">
            <div>
              <h3 className="best-param-modal__title">Tuned Hyperparameter Presets</h3>
              <p className="best-param-modal__subtitle">
                Review hyperparameter sets discovered by recent tuning runs and apply them to the current model.
              </p>
            </div>
            <button type="button" className="best-param-modal__close" onClick={onClose}>
              Close
            </button>
          </header>
          <div className="best-param-modal__layout">
            <aside className="best-param-modal__list" aria-label="Tuned presets">
              {presets.map((preset) => {
                const isActive = preset.id === selectedPreset?.id;
                const presetRelative = preset.finishedAt ? formatRelativeTime(preset.finishedAt) : null;
                return (
                  <button
                    key={preset.id}
                    type="button"
                    className={`best-param-modal__list-button${isActive ? ' best-param-modal__list-button--active' : ''}`}
                    onClick={() => setSelectedId(preset.id)}
                  >
                    <span className="best-param-modal__list-title">{preset.label}</span>
                    <span className="best-param-modal__list-meta">
                      {preset.runNumber ? `Run ${preset.runNumber}` : 'Tuning run'}
                      {presetRelative ? ` • ${presetRelative}` : ''}
                    </span>
                    {preset.scoring && (
                      <span className="best-param-modal__list-score">
                        {preset.scoring}: {formatPresetScore(preset.score)}
                      </span>
                    )}
                    {preset.targetColumn && (
                      <span className="best-param-modal__list-target">Target: {preset.targetColumn}</span>
                    )}
                  </button>
                );
              })}
            </aside>
            <section className="best-param-modal__details">
              {selectedPreset ? (
                <>
                  <div className="best-param-modal__summary">
                    <div className="best-param-modal__summary-main">
                      <h4>{selectedPreset.label}</h4>
                      <p>
                        {selectedPreset.runNumber ? `Run ${selectedPreset.runNumber}` : 'Tuning run'}
                        {relativeFinished ? ` • ${relativeFinished}` : ''}
                        {selectedPreset.searchStrategy ? ` • ${selectedPreset.searchStrategy}` : ''}
                      </p>
                      <p>
                        Model: <strong>{selectedPreset.modelType}</strong>
                        {selectedPreset.targetColumn ? (
                          <>
                            {' '}
                            • Target: <strong>{selectedPreset.targetColumn}</strong>
                          </>
                        ) : null}
                        {selectedPreset.scoring ? (
                          <>
                            {' '}
                            • {selectedPreset.scoring}: <strong>{formatPresetScore(selectedPreset.score)}</strong>
                          </>
                        ) : null}
                      </p>
                    </div>
                    <div className="best-param-modal__summary-actions">
                      <button
                        type="button"
                        className="best-param-modal__apply"
                        onClick={() => onApply(selectedPreset)}
                        disabled={!paramEntries.length}
                      >
                        {currentIsApplied ? 'Already applied' : 'Apply preset'}
                      </button>
                    </div>
                  </div>
                  {selectedPreset.description && (
                    <p className="best-param-modal__note">{selectedPreset.description}</p>
                  )}
                  {scalingNotice && (
                    <div
                      className="best-param-modal__notice best-param-modal__notice--warning"
                      style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem', margin: '0.5rem 0' }}
                    >
                      <strong>Tip:</strong> {scalingNotice}
                    </div>
                  )}
                  <div className="best-param-modal__param-grid" role="table">
                    {paramEntries.length ? (
                      paramEntries.map((entry) => (
                        <div key={entry.key} className="best-param-modal__param-row" role="row">
                          <div className="best-param-modal__param-name" role="cell">
                            <div className="best-param-modal__param-label">
                              {entry.label}
                              {entry.label !== entry.key && (
                                <span className="best-param-modal__param-key">
                                  <code>{entry.key}</code>
                                </span>
                              )}
                            </div>
                            {entry.description && (
                              <div className="best-param-modal__param-description">{entry.description}</div>
                            )}
                          </div>
                          <div className="best-param-modal__param-value" role="cell">
                            <code>{entry.value}</code>
                            {entry.note && (
                              <span className="best-param-modal__param-note">{entry.note}</span>
                            )}
                          </div>
                        </div>
                      ))
                    ) : (
                      <p className="best-param-modal__empty">No hyperparameters available for this preset.</p>
                    )}
                  </div>
                </>
              ) : (
                <p className="best-param-modal__empty">Select a tuning run to inspect its hyperparameters.</p>
              )}
            </section>
          </div>
        </div>
      </div>
    );
  }, [activePresetId, currentIsApplied, fieldMetadata, isOpen, onApply, onClose, presets, selectedPreset]);

  if (!content) {
    return null;
  }

  return createPortal(content, document.body);
};
