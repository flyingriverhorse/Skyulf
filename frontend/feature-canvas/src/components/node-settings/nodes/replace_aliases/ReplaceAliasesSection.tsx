import React, { useCallback, useEffect, useMemo, useState } from 'react';
import type { FeatureNodeParameter } from '../../../../api';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import { extractPendingConfigurationDetails, type PendingConfigurationDetail } from '../../utils/pendingConfiguration';
import {
  ALIAS_MODE_OPTIONS,
  type AliasColumnOption,
  type AliasColumnSummary,
  type AliasCustomPairSummary,
  type AliasMode,
  type AliasModeOption,
  type AliasSampleMap,
  type AliasStrategyConfig,
  getAliasModeDetails,
  summarizeAliasSamples,
  summarizeNonTextColumns,
  summarizeRecommendedColumns,
} from './replaceAliasesSettings';

const normalizePreviewToken = (value: string): string => value.replace(/[^0-9a-zA-Z]+/g, '').toLowerCase();

const COUNTRY_ALIAS_PREVIEW: Array<[string, string]> = [
  ['UK', 'United Kingdom'],
  ['UAE', 'United Arab Emirates'],
  ['USA', 'United States'],
  ['PRC', 'China'],
  ['KSA', 'Saudi Arabia'],
  ['Brasil', 'Brazil'],
];

const COUNTRY_ALIAS_HINTS = new Set<string>([
  'us',
  'usa',
  'unitedstates',
  'unitedstatesofamerica',
  'uk',
  'unitedkingdom',
  'greatbritain',
  'england',
  'uae',
  'unitedarabemirates',
  'prc',
  'china',
  'southkorea',
  'republicofkorea',
  'ksa',
  'saudiarabia',
  'brasil',
  'brazil',
  'de',
  'germany',
  'fr',
  'france',
  'es',
  'spain',
  'ca',
  'canada',
  'mx',
  'mexico',
  'country',
  'state',
  'province',
]);

COUNTRY_ALIAS_PREVIEW.forEach((pair) => {
  pair.forEach((token) => COUNTRY_ALIAS_HINTS.add(normalizePreviewToken(token)));
});

const BOOLEAN_ALIAS_PREVIEW: Array<[string, string]> = [
  ['Y', 'Yes'],
  ['1', 'Yes'],
  ['ON', 'Yes'],
  ['N', 'No'],
  ['0', 'No'],
  ['OFF', 'No'],
];

const BOOLEAN_ALIAS_HINTS = new Set<string>([
  'yes',
  'y',
  'true',
  't',
  '1',
  'on',
  'enable',
  'enabled',
  'no',
  'n',
  'false',
  'f',
  '0',
  'off',
  'disable',
  'disabled',
  'active',
  'inactive',
  'flag',
]);

BOOLEAN_ALIAS_PREVIEW.forEach((pair) => {
  pair.forEach((token) => BOOLEAN_ALIAS_HINTS.add(normalizePreviewToken(token)));
});

const PUNCTUATION_ALIAS_PREVIEW: Array<[string, string]> = [
  ['C.R.M.', 'CRM'],
  ['Part - Time', 'Part Time'],
  ['end-to-end', 'end to end'],
];

const COUNTRY_CODE_REGEX = /^[A-Za-z]{2,3}$/;
const PUNCTUATION_REGEX = /[^0-9A-Za-z\s]/;
const TOKEN_SPLIT_REGEX = /[^0-9A-Za-z]+/;

const MODE_REFERENCE_INFO: Record<
  AliasMode,
  {
    title: string;
    description: string;
    preview?: Array<[string, string]>;
  }
> = {
  canonicalize_country_codes: {
    title: 'Country alias map',
    description: 'Common country abbreviations resolve to canonical names or ISO codes.',
    preview: COUNTRY_ALIAS_PREVIEW,
  },
  normalize_boolean: {
    title: 'Boolean token map',
    description: 'Typical yes / no style values are standardized.',
    preview: BOOLEAN_ALIAS_PREVIEW,
  },
  punctuation: {
    title: 'Punctuation cleanup',
    description: 'Removes punctuation and condenses whitespace for easier matching.',
    preview: PUNCTUATION_ALIAS_PREVIEW,
  },
  custom: {
    title: 'Custom mappings',
    description: 'Define your own alias => replacement pairs below.',
  },
};

const collectTokens = (samples: string[]): string[] => {
  const tokens: string[] = [];
  samples.forEach((sample) => {
    const pieces = sample.split(TOKEN_SPLIT_REGEX).filter(Boolean);
    pieces.forEach((piece) => {
      tokens.push(piece);
    });
  });
  return tokens;
};

const hasModeAffinity = (mode: AliasMode, samples: string[]): boolean => {
  if (!samples.length) {
    return false;
  }
  if (mode === 'punctuation') {
    return samples.some((value) => PUNCTUATION_REGEX.test(value));
  }

  const tokens = collectTokens(samples).map((token) => normalizePreviewToken(token));
  if (!tokens.length) {
    return false;
  }

  if (mode === 'canonicalize_country_codes') {
    return tokens.some((token) => COUNTRY_ALIAS_HINTS.has(token)) || samples.some((sample) => COUNTRY_CODE_REGEX.test(sample.trim()));
  }

  if (mode === 'normalize_boolean') {
    return tokens.some((token) => BOOLEAN_ALIAS_HINTS.has(token));
  }

  return false;
};

export type ReplaceAliasesSectionProps = {
  hasSource: boolean;
  hasReachableSource: boolean;
  strategies: AliasStrategyConfig[];
  columnSummary: AliasColumnSummary;
  columnOptions: AliasColumnOption[];
  sampleMap: AliasSampleMap;
  modeOptions?: AliasModeOption[];
  customPairSummary: AliasCustomPairSummary;
  customPairsParameter: FeatureNodeParameter | null;
  collapsedStrategies: Set<number>;
  onToggleStrategy: (index: number) => void;
  onRemoveStrategy: (index: number) => void;
  onAddStrategy: () => void;
  onModeChange: (index: number, mode: AliasMode) => void;
  onAutoDetectToggle: (index: number, enabled: boolean) => void;
  onColumnToggle: (index: number, column: string) => void;
  onColumnsChange: (index: number, value: string) => void;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  previewState?: PreviewState;
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
  onPendingConfigurationCleared?: () => void;
};

const formatSamplePreview = (samples: string[]): string | null => {
  if (!samples.length) {
    return null;
  }
  const preview = samples.slice(0, 3);
  const overflow = samples.length - preview.length;
  return overflow > 0 ? `${preview.join(', ')} ...` : preview.join(', ');
};

export const ReplaceAliasesSection: React.FC<ReplaceAliasesSectionProps> = ({
  hasSource,
  hasReachableSource,
  strategies,
  columnSummary,
  columnOptions,
  sampleMap,
  modeOptions = ALIAS_MODE_OPTIONS,
  customPairSummary,
  customPairsParameter,
  collapsedStrategies,
  onToggleStrategy,
  onRemoveStrategy,
  onAddStrategy,
  onModeChange,
  onAutoDetectToggle,
  onColumnToggle,
  onColumnsChange,
  renderParameterField,
  previewState,
  onPendingConfigurationWarning,
  onPendingConfigurationCleared,
}) => {
  useEffect(() => {
    if (previewState?.data?.signals?.full_execution) {
      const details = extractPendingConfigurationDetails(previewState.data.signals.full_execution);
      if (details.length > 0) {
        onPendingConfigurationWarning?.(details);
      } else {
        onPendingConfigurationCleared?.();
      }
    }
  }, [previewState, onPendingConfigurationWarning, onPendingConfigurationCleared]);

  const hasStrategies = strategies.length > 0;
  const selectedCount = columnSummary.selectedColumns.length;
  const recommendedSummary = useMemo(
    () => summarizeRecommendedColumns(columnSummary.recommendedColumns),
    [columnSummary.recommendedColumns],
  );
  const nonTextSummary = useMemo(
    () => summarizeNonTextColumns(columnSummary.nonTextSelected),
    [columnSummary.nonTextSelected],
  );
  const samplePreviews = useMemo(
    () => summarizeAliasSamples(sampleMap, columnSummary.selectedColumns),
    [columnSummary.selectedColumns, sampleMap],
  );
  const disconnectedWarning = !hasSource || !hasReachableSource;

  const duplicateCount = customPairSummary.duplicates.length + customPairSummary.duplicateOverflow;
  const invalidCount = customPairSummary.invalidEntries.length + customPairSummary.invalidOverflow;
  const duplicatePreviewText = customPairSummary.duplicates.join(', ');
  const invalidPreviewText = customPairSummary.invalidEntries.join(', ');

  const hasCustomStrategy = useMemo(
    () => strategies.some((strategy) => strategy.mode === 'custom'),
    [strategies],
  );

  const columnUsage = useMemo(() => {
    const usage = new Map<string, number>();
    strategies.forEach((strategy) => {
      strategy.columns.forEach((column) => {
        const normalized = column.trim();
        if (!normalized) {
          return;
        }
        usage.set(normalized, (usage.get(normalized) ?? 0) + 1);
      });
    });
    return usage;
  }, [strategies]);

  const [expandedColumnLists, setExpandedColumnLists] = useState<Set<number>>(() => new Set());

  const toggleAdditionalColumns = useCallback((index: number) => {
    setExpandedColumnLists((previous) => {
      const next = new Set(previous);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }, []);

  useEffect(() => {
    setExpandedColumnLists((previous) => {
      const next = new Set<number>();
      previous.forEach((value) => {
        if (value < strategies.length) {
          next.add(value);
        }
      });
      return next.size === previous.size ? previous : next;
    });
  }, [strategies.length]);

  const [modePreview, setModePreview] = useState<{ index: number | null; mode: AliasMode | null }>(
    () => ({ index: null, mode: null }),
  );

  const columnSamples = useMemo(() => {
    const cache = new Map<string, string[]>();
    columnOptions.forEach((option) => {
      const baseSamples = Array.isArray(option.samples) ? option.samples : [];
      const extended = Array.isArray(sampleMap[option.name]) ? sampleMap[option.name] : [];
      const combined = [...baseSamples, ...extended]
        .map((entry) => String(entry ?? '').trim())
        .filter((entry) => entry.length > 0);
      cache.set(option.name, combined.slice(0, 12));
    });
    return cache;
  }, [columnOptions, sampleMap]);

  const modeColumnAffinity = useMemo(() => {
    const affinity = new Map<AliasMode, Set<string>>();
    columnOptions.forEach((option) => {
      const samples = columnSamples.get(option.name) ?? [];
      const inspectionSamples = samples.length ? samples : [option.name];
      (['canonicalize_country_codes', 'normalize_boolean', 'punctuation'] as AliasMode[]).forEach((mode) => {
        if (hasModeAffinity(mode, inspectionSamples)) {
          if (!affinity.has(mode)) {
            affinity.set(mode, new Set());
          }
          affinity.get(mode)!.add(option.name);
        }
      });
    });
    return affinity;
  }, [columnOptions, columnSamples]);

  useEffect(() => {
    if (modePreview.index === null) {
      return;
    }
    const active = strategies[modePreview.index];
    if (!active || active.mode !== modePreview.mode) {
      setModePreview({ index: null, mode: null });
    }
  }, [modePreview.index, modePreview.mode, strategies]);

  const handleOpenModePreview = useCallback((index: number, mode: AliasMode) => {
    setModePreview({ index, mode });
  }, []);

  const handleCloseModePreview = useCallback(() => {
    setModePreview({ index: null, mode: null });
  }, []);

  let customPairsRendered = false;

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Alias cleanup strategies</h3>
        <div className="canvas-modal__section-actions">
          <button type="button" className="btn btn-outline-secondary" onClick={onAddStrategy}>
            Add strategy
          </button>
        </div>
      </div>

      {columnSummary.autoDetectionActive ? (
        <p className="canvas-modal__note">
          Auto-detect is enabled; any remaining text-like columns will be swept into the strategies when this node runs.
        </p>
      ) : (
        <p className="canvas-modal__note">
          Targeting <strong>{selectedCount}</strong> column{selectedCount === 1 ? '' : 's'} across all strategies.
        </p>
      )}

      {samplePreviews.length > 0 && (
        <p className="canvas-modal__note">
          Samples: {samplePreviews.map((preview) => `${preview.column} (${preview.values.join(', ')})`).join('; ')}.
        </p>
      )}

      {columnSummary.textColumnCount > 0 && recommendedSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Other text columns: {recommendedSummary.preview.join(', ')}
          {recommendedSummary.remaining > 0 ? `, ... (${recommendedSummary.remaining} more)` : ''}.
        </p>
      )}

      {nonTextSummary.preview.length > 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          {nonTextSummary.preview.join(', ')} {nonTextSummary.preview.length === 1 ? 'is' : 'are'} not text-like.
          {nonTextSummary.remaining > 0
            ? ` ${nonTextSummary.remaining} more column${nonTextSummary.remaining === 1 ? ' is' : 's are'} also non-text.`
            : ' Remove them or double-check the data.'}
        </p>
      )}

      {disconnectedWarning ? (
        !hasSource ? (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Select a dataset to load available columns for alias cleanup.
          </p>
        ) : (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Connect this node to an upstream output to inspect available columns.
          </p>
        )
      ) : null}

      {!hasStrategies && (
        <p className="canvas-modal__note">
          No strategies configured yet. Combine presets (countries, booleans, punctuation) or add custom maps to standardize aliases.
        </p>
      )}

      {hasStrategies && (
        <div className="canvas-imputer__list">
          {strategies.map((strategy, index) => {
            const modeDetails = getAliasModeDetails(strategy.mode);
            const modeLabel =
              modeOptions.find((option) => option.value === strategy.mode)?.label ?? modeDetails.label;
            const columnCount = strategy.columns.length;
            const columnSummaryLabel = columnCount
              ? `${columnCount} column${columnCount === 1 ? '' : 's'}`
              : strategy.autoDetect
                ? 'Auto-detecting columns'
                : 'No columns yet';
            const isCollapsed = collapsedStrategies.has(index);
            const isOtherExpanded = expandedColumnLists.has(index);
            const columnsValue = strategy.columns.join(', ');
            const affinityColumns = modeColumnAffinity.get(strategy.mode) ?? new Set<string>();
            const showCustomPairsField =
              strategy.mode === 'custom' && Boolean(customPairsParameter) && !customPairsRendered;
            const showCustomPairsNotice =
              strategy.mode === 'custom' && Boolean(customPairsParameter) && customPairsRendered;

            if (showCustomPairsField) {
              customPairsRendered = true;
            }

            const sortedOptions = columnOptions
              .map((option) => {
                const isSelected = strategy.columns.includes(option.name);
                const hasAffinity = affinityColumns.has(option.name);
                const priority = isSelected
                  ? 0
                  : hasAffinity
                    ? 1
                    : option.isRecommended
                      ? 2
                      : option.isTextLike
                        ? 3
                        : 4;
                return { option, priority };
              })
              .sort((a, b) => {
                if (a.priority !== b.priority) {
                  return a.priority - b.priority;
                }
                return a.option.name.localeCompare(b.option.name);
              })
              .map((entry) => entry.option);

            const hasAffinitySuggestions = affinityColumns.size > 0;
            const primaryOptions = sortedOptions.filter((option) => {
              if (strategy.columns.includes(option.name)) {
                return true;
              }
              if (hasAffinitySuggestions) {
                return affinityColumns.has(option.name);
              }
              return option.isRecommended || option.isTextLike;
            });

            const primarySet = new Set(primaryOptions.map((option) => option.name));
            const otherOptions = sortedOptions.filter((option) => !primarySet.has(option.name));

            return (
              <div key={`alias-strategy-${index}`} className="canvas-imputer__card">
                <div className="canvas-imputer__card-header">
                  <button
                    type="button"
                    className="canvas-imputer__card-toggle"
                    onClick={() => onToggleStrategy(index)}
                    aria-expanded={!isCollapsed}
                    aria-controls={`alias-strategy-body-${index}`}
                  >
                    <span
                      className={`canvas-imputer__toggle-icon${
                        isCollapsed ? '' : ' canvas-imputer__toggle-icon--open'
                      }`}
                      aria-hidden="true"
                    />
                    <span className="canvas-imputer__card-text">
                      <span className="canvas-imputer__card-title">Strategy {index + 1}</span>
                      <span className="canvas-imputer__card-summary">
                        {modeLabel}
                        {strategy.autoDetect ? ' · auto-detect' : ''}
                        {columnSummaryLabel ? ` · ${columnSummaryLabel}` : ''}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    className="canvas-imputer__remove"
                    onClick={() => onRemoveStrategy(index)}
                    aria-label={`Remove alias strategy ${index + 1}`}
                  >
                    Remove
                  </button>
                </div>
                {!isCollapsed && (
                  <div className="canvas-imputer__card-body" id={`alias-strategy-body-${index}`}>
                    <div className="canvas-imputer__row">
                      <label>Replacement strategy</label>
                      <div className="canvas-imputer__mode-toggle" role="radiogroup" aria-label="Alias replacement strategy">
                        {modeOptions.map((option) => {
                          const isActive = option.value === strategy.mode;
                          return (
                            <button
                              key={option.value}
                              type="button"
                              className={`btn btn-outline-secondary canvas-imputer__mode-button${
                                isActive ? ' canvas-imputer__mode-button--active active' : ''
                              }`}
                              role="radio"
                              aria-checked={isActive}
                              onClick={() => onModeChange(index, option.value)}
                            >
                              {option.label}
                            </button>
                          );
                        })}
                        <button
                          type="button"
                          className="btn btn-link canvas-imputer__mode-info"
                          onClick={() => handleOpenModePreview(index, strategy.mode)}
                        >
                          View examples
                        </button>
                      </div>
                      <p className="canvas-modal__meta">{modeDetails.guidance}</p>
                    </div>

                    {modePreview.index === index && modePreview.mode === strategy.mode && (
                      <div className="canvas-modal__note canvas-modal__note--info">
                        <div>
                          <strong>{MODE_REFERENCE_INFO[strategy.mode].title}</strong>
                          <button type="button" className="btn btn-link" onClick={handleCloseModePreview}>
                            Close
                          </button>
                        </div>
                        <p>{MODE_REFERENCE_INFO[strategy.mode].description}</p>
                        {MODE_REFERENCE_INFO[strategy.mode].preview && (
                          <ul>
                            {MODE_REFERENCE_INFO[strategy.mode].preview!.map(([alias, mapped]) => (
                              <li key={`${alias}-${mapped}`}>
                                {alias} <span aria-hidden="true">-&gt;</span> {mapped}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    )}

                    <div className="canvas-imputer__row">
                      <label htmlFor={`alias-strategy-${index}-auto-detect`}>Detection</label>
                      <div className="canvas-modal__boolean-control">
                        <input
                          id={`alias-strategy-${index}-auto-detect`}
                          type="checkbox"
                          checked={strategy.autoDetect}
                          onChange={(event) => onAutoDetectToggle(index, event.target.checked)}
                        />
                        <label htmlFor={`alias-strategy-${index}-auto-detect`}>
                          Auto-detect additional text columns
                        </label>
                      </div>
                      <p className="canvas-modal__note">
                        Detected text-like columns not assigned elsewhere will be appended to this strategy.
                      </p>
                    </div>

                    <div className="canvas-imputer__row">
                      <label>Columns</label>
                      {columnOptions.length ? (
                        <>
                          {affinityColumns.size > 0 && (
                            <p className="canvas-modal__note">
                              Highlighting columns that resemble {modeLabel.toLowerCase()} entries.
                            </p>
                          )}
                          {primaryOptions.length ? (
                            <div
                              className="canvas-imputer__columns-list"
                              role="listbox"
                              aria-label={`Columns assigned to alias strategy ${index + 1}`}
                            >
                              {primaryOptions.map((option) => {
                                const isSelected = strategy.columns.includes(option.name);
                                const preview = formatSamplePreview(
                                  option.samples.length ? option.samples : sampleMap[option.name] ?? [],
                                );
                                const usageCount = columnUsage.get(option.name) ?? 0;
                                const assignedElsewhere = usageCount - (isSelected ? 1 : 0) > 0;
                                const hasAffinity = affinityColumns.has(option.name);
                                return (
                                  <button
                                    key={option.name}
                                    type="button"
                                    className={`canvas-imputer__column-pill${
                                      isSelected ? ' canvas-imputer__column-pill--selected' : ''
                                    }`}
                                    onClick={() => onColumnToggle(index, option.name)}
                                    aria-pressed={isSelected}
                                  >
                                    <span className="canvas-imputer__column-pill-name">{option.name}</span>
                                    <div className="canvas-imputer__column-pill-meta">
                                      {option.dtype && (
                                        <span className="canvas-imputer__column-pill-dtype">{option.dtype}</span>
                                      )}
                                      {option.isRecommended && <span className="canvas-modal__meta">Suggested</span>}
                                      {!option.isTextLike && <span className="canvas-modal__meta">Non-text</span>}
                                      {hasAffinity && <span className="canvas-modal__meta">Likely match</span>}
                                      {assignedElsewhere && <span className="canvas-modal__meta">Used elsewhere</span>}
                                    </div>
                                    {preview && <span className="canvas-modal__meta">e.g. {preview}</span>}
                                  </button>
                                );
                              })}
                            </div>
                          ) : (
                            <p className="canvas-modal__note">
                              No detected or selected columns yet. Expand other columns below to assign manually.
                            </p>
                          )}

                          {otherOptions.length > 0 && (
                            <div className="canvas-imputer__more-columns">
                              <button
                                type="button"
                                className="btn btn-outline-secondary"
                                onClick={() => toggleAdditionalColumns(index)}
                                aria-expanded={isOtherExpanded}
                                aria-controls={`alias-strategy-${index}-other-columns`}
                              >
                                {isOtherExpanded
                                  ? 'Hide other columns'
                                  : `Show other columns (${otherOptions.length})`}
                              </button>
                              {isOtherExpanded && (
                                <div
                                  id={`alias-strategy-${index}-other-columns`}
                                  className="canvas-imputer__columns-list"
                                  role="listbox"
                                  aria-label={`Other available columns for alias strategy ${index + 1}`}
                                >
                                  {otherOptions.map((option) => {
                                    const isSelected = strategy.columns.includes(option.name);
                                    const preview = formatSamplePreview(
                                      option.samples.length ? option.samples : sampleMap[option.name] ?? [],
                                    );
                                    const usageCount = columnUsage.get(option.name) ?? 0;
                                    const assignedElsewhere = usageCount - (isSelected ? 1 : 0) > 0;
                                    const hasAffinity = affinityColumns.has(option.name);
                                    return (
                                      <button
                                        key={option.name}
                                        type="button"
                                        className={`canvas-imputer__column-pill${
                                          isSelected ? ' canvas-imputer__column-pill--selected' : ''
                                        }`}
                                        onClick={() => onColumnToggle(index, option.name)}
                                        aria-pressed={isSelected}
                                      >
                                        <span className="canvas-imputer__column-pill-name">{option.name}</span>
                                        <div className="canvas-imputer__column-pill-meta">
                                          {option.dtype && (
                                            <span className="canvas-imputer__column-pill-dtype">{option.dtype}</span>
                                          )}
                                          {!option.isTextLike && <span className="canvas-modal__meta">Non-text</span>}
                                          {hasAffinity && <span className="canvas-modal__meta">Likely match</span>}
                                          {assignedElsewhere && <span className="canvas-modal__meta">Used elsewhere</span>}
                                        </div>
                                        {preview && <span className="canvas-modal__meta">e.g. {preview}</span>}
                                      </button>
                                    );
                                  })}
                                </div>
                              )}
                            </div>
                          )}
                        </>
                      ) : (
                        <p className="canvas-modal__note">
                          No catalogued columns yet. Add column names manually below or enable auto-detect to populate this list.
                        </p>
                      )}
                    </div>

                    <div
                      className={`canvas-imputer__selected${
                        strategy.columns.length ? '' : ' canvas-imputer__selected--empty'
                      }`}
                      aria-live="polite"
                    >
                      {strategy.columns.length ? (
                        strategy.columns.map((column) => (
                          <span key={column} className="canvas-imputer__selected-chip">
                            {column}
                          </span>
                        ))
                      ) : (
                        <span>No manual columns selected yet.</span>
                      )}
                    </div>

                    <div className="canvas-imputer__row">
                      <label htmlFor={`alias-strategy-${index}-manual-columns`}>
                        Manual columns (optional)
                      </label>
                      <input
                        id={`alias-strategy-${index}-manual-columns`}
                        type="text"
                        className="canvas-imputer__columns-input"
                        value={columnsValue}
                        onChange={(event) => onColumnsChange(index, event.target.value)}
                        placeholder="Add custom columns, separated by commas"
                      />
                    </div>

                    <p className="canvas-modal__note">
                      Tap suggestions to toggle columns. Manual entries let you set custom fields that are not listed above.
                    </p>

                    {strategy.mode === 'custom' && customPairsParameter && (
                      <div className="canvas-imputer__row">
                        <label>Custom pairs</label>
                        <p className="canvas-modal__meta">
                          Enter one alias =&gt; replacement per line. These mappings apply to every custom strategy in this node.
                        </p>
                        <div className="canvas-modal__parameter-list">
                          {showCustomPairsField
                            ? renderParameterField(customPairsParameter)
                            : showCustomPairsNotice
                              ? (
                                  <p className="canvas-modal__note">
                                    Edit the custom pairs above to update this strategy.
                                  </p>
                                )
                              : null}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {customPairSummary.totalPairs > 0 && (
        <p className="canvas-modal__note">
          {customPairSummary.totalPairs} custom pair{customPairSummary.totalPairs === 1 ? '' : 's'} captured.
          {hasCustomStrategy
            ? ' They will be applied during execution.'
            : ' Switch a strategy to Custom mappings to apply them.'}
          {customPairSummary.previewPairs.length > 0
            ? ` Examples: ${customPairSummary.previewPairs.join('; ')}${
                customPairSummary.previewOverflow > 0 ? '...' : ''
              }`
            : ''}
        </p>
      )}

      {customPairSummary.totalPairs === 0 && hasCustomStrategy && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          At least one strategy is set to Custom mappings, but no valid alias pairs were detected.
        </p>
      )}

      {duplicateCount > 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Duplicate alias entries{duplicatePreviewText ? `: ${duplicatePreviewText}` : ''}
          {customPairSummary.duplicateOverflow > 0 ? '...' : ''}. Only the first entry for each alias will be used.
        </p>
      )}

      {invalidCount > 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Ignored {invalidCount} incomplete pair{invalidCount === 1 ? '' : 's'}.
          {invalidPreviewText
            ? ` Examples: ${invalidPreviewText}${customPairSummary.invalidOverflow > 0 ? '...' : ''}`
            : ''}
        </p>
      )}
    </section>
  );
};
