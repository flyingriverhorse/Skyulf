import React, { ChangeEvent, useEffect, useMemo, useState } from 'react';
import type {
  FeatureMathNodeSignal,
  FeatureMathOperationStatus,
  FeatureNodeParameter,
} from '../../../../api';
import {
  DATETIME_FEATURE_OPTIONS,
  FEATURE_MATH_TYPE_OPTIONS,
  FeatureMathOperationDraft,
  FeatureMathOperationSummary,
  FeatureMathOperationType,
  getMethodOptions,
  getSimilarityMethodDescription,
} from './featureMathSettings';

const STATUS_LABELS: Record<FeatureMathOperationStatus, string> = {
  applied: 'Applied',
  skipped: 'Skipped',
  failed: 'Failed',
};

const STATUS_CHIP_CLASS: Record<FeatureMathOperationStatus, string> = {
  applied: 'canvas-cast__chip--applied',
  skipped: 'canvas-cast__chip--muted',
  failed: 'canvas-cast__chip--attention',
};

const DATETIME_FEATURE_KEY_SET = new Set(DATETIME_FEATURE_OPTIONS.map((option) => option.value));

const formatColumnPreview = (columns: string[], limit = 4): string => {
  if (!columns.length) {
    return 'No columns yet';
  }
  const preview = columns.slice(0, limit);
  const remainder = columns.length - preview.length;
  return remainder > 0 ? `${preview.join(', ')} … (${remainder} more)` : preview.join(', ');
};

const parseColumnList = (value: string): string[] => {
  if (!value.trim()) {
    return [];
  }
  return Array.from(
    new Set(
      value
        .split(',')
        .map((segment) => segment.trim())
        .filter((segment) => segment.length > 0),
    ),
  ).sort((a, b) => a.localeCompare(b));
};

const parseConstantsList = (value: string): number[] => {
  if (!value.trim()) {
    return [];
  }
  return value
    .split(',')
    .map((segment) => Number(segment.trim()))
    .filter((numeric) => Number.isFinite(numeric));
};

const getSummaryNoteClass = (status: FeatureMathOperationStatus | undefined): string => {
  if (status === 'failed') {
    return 'canvas-modal__note canvas-modal__note--error';
  }
  if (status === 'skipped') {
    return 'canvas-modal__note canvas-modal__note--warning';
  }
  return 'canvas-modal__note canvas-modal__note--info';
};

type FeatureMathOperationCardProps = {
  index: number;
  total: number;
  operation: FeatureMathOperationDraft;
  summary?: FeatureMathOperationSummary;
  availableColumns: string[];
  isCollapsed: boolean;
  onToggleCollapsed: (operationId: string) => void;
  onDuplicate: (operationId: string) => void;
  onRemove: (operationId: string) => void;
  onReorder: (operationId: string, direction: 'up' | 'down') => void;
  onChange: (operationId: string, updates: Partial<FeatureMathOperationDraft>) => void;
};

const FeatureMathOperationCard: React.FC<FeatureMathOperationCardProps> = ({
  index,
  total,
  operation,
  summary,
  availableColumns,
  isCollapsed,
  onToggleCollapsed,
  onDuplicate,
  onRemove,
  onReorder,
  onChange,
}) => {
  const methodOptions = useMemo(() => getMethodOptions(operation.type), [operation.type]);
  const typeLabel = useMemo(
    () => FEATURE_MATH_TYPE_OPTIONS.find((option) => option.value === operation.type)?.label ?? operation.type,
    [operation.type],
  );
  const methodLabel = useMemo(
    () => methodOptions.find((option) => option.value === operation.method)?.label ?? operation.method,
    [methodOptions, operation.method],
  );
  const similarityDescription = useMemo(
    () => (operation.type === 'similarity' ? getSimilarityMethodDescription(operation.method) : ''),
    [operation.method, operation.type],
  );

  const [inputColumnsValue, setInputColumnsValue] = useState(operation.inputColumns.join(', '));
  const [secondaryColumnsValue, setSecondaryColumnsValue] = useState(operation.secondaryColumns.join(', '));
  const [constantsValue, setConstantsValue] = useState(operation.constants.join(', '));

  useEffect(() => {
    setInputColumnsValue(operation.inputColumns.join(', '));
  }, [operation.inputColumns]);

  useEffect(() => {
    setSecondaryColumnsValue(operation.secondaryColumns.join(', '));
  }, [operation.secondaryColumns]);

  useEffect(() => {
    setConstantsValue(operation.constants.join(', '));
  }, [operation.constants]);

  const handleInputColumnsCommit = () => {
    onChange(operation.id, { inputColumns: parseColumnList(inputColumnsValue) });
  };

  const handleSecondaryColumnsCommit = () => {
    onChange(operation.id, { secondaryColumns: parseColumnList(secondaryColumnsValue) });
  };

  const handleConstantsCommit = () => {
    onChange(operation.id, { constants: parseConstantsList(constantsValue) });
  };

  const toggleId = `feature-math-toggle-${operation.id}`;
  const bodyId = `feature-math-body-${operation.id}`;
  const inputListId = `feature-math-input-columns-${operation.id}`;
  const secondaryListId = `feature-math-secondary-columns-${operation.id}`;

  const chipStatus = summary?.status;
  const statusChip = chipStatus
    ? (
        <span
          className={`canvas-cast__chip ${STATUS_CHIP_CLASS[chipStatus] ?? ''}`}
          style={{ marginLeft: '0.5rem' }}
        >
          {STATUS_LABELS[chipStatus]}
        </span>
      )
    : null;

  const summaryMessage = summary?.message ? summary.message.trim() : '';
  const summaryNoteClass = getSummaryNoteClass(chipStatus);
  const outputColumnsPreview = summary?.outputColumns?.length ? summary.outputColumns.join(', ') : '';
  const columnsPreview = formatColumnPreview(operation.inputColumns);

  const showSecondaryField = operation.type === 'ratio' || operation.type === 'similarity';
  const showConstantsField = operation.type === 'arithmetic' || operation.type === 'stat' || operation.type === 'ratio';
  const showNumericTuning = operation.type === 'arithmetic' || operation.type === 'stat' || operation.type === 'ratio';
  const showSimilarityOptions = operation.type === 'similarity';
  const showDatetimeFields = operation.type === 'datetime_extract';
  const showPrefixField = operation.type === 'stat' || operation.type === 'datetime_extract';

  const availableColumnsPreview = useMemo(() => {
    if (!availableColumns.length) {
      return '';
    }
    const preview = availableColumns.slice(0, 6);
    const remainder = availableColumns.length - preview.length;
    return remainder > 0 ? `${preview.join(', ')} … (${remainder} more)` : preview.join(', ');
  }, [availableColumns]);

  const handleDateFeatureToggle = (feature: string) => {
    if (!feature || !DATETIME_FEATURE_KEY_SET.has(feature)) {
      return;
    }
    const next = new Set(operation.datetimeFeatures);
    if (next.has(feature)) {
      next.delete(feature);
    } else {
      next.add(feature);
    }
    onChange(operation.id, { datetimeFeatures: Array.from(next) });
  };

  const handleNumberChange = (key: 'fillna' | 'roundDigits' | 'epsilon') => (event: ChangeEvent<HTMLInputElement>) => {
    const raw = event.target.value;
    if (raw === '') {
      onChange(operation.id, { [key]: null } as Partial<FeatureMathOperationDraft>);
      return;
    }
    const numeric = Number(raw);
    if (!Number.isFinite(numeric)) {
      return;
    }
    if (key === 'roundDigits') {
      onChange(operation.id, { roundDigits: Math.round(numeric) });
    } else if (key === 'fillna') {
      onChange(operation.id, { fillna: numeric });
    } else {
      onChange(operation.id, { epsilon: numeric });
    }
  };

  const allowOverwriteValue = operation.allowOverwrite === null
    ? 'inherit'
    : operation.allowOverwrite
      ? 'allow'
      : 'prevent';

  return (
    <div className="canvas-imputer__card">
      <div className="canvas-imputer__card-header">
        <button
          type="button"
          id={toggleId}
          className="canvas-imputer__card-toggle"
          onClick={() => onToggleCollapsed(operation.id)}
          aria-expanded={!isCollapsed}
          aria-controls={bodyId}
        >
          <span
            className={`canvas-imputer__toggle-icon${isCollapsed ? '' : ' canvas-imputer__toggle-icon--open'}`}
            aria-hidden="true"
          />
          <span className="canvas-imputer__card-text">
            <span className="canvas-imputer__card-title">Operation {index + 1}</span>
            <span className="canvas-imputer__card-summary">
              {typeLabel}
              {methodLabel ? ` · ${methodLabel}` : ''}
              {operation.outputColumn ? ` · → ${operation.outputColumn}` : ''}
              {statusChip}
            </span>
            <span className="canvas-modal__meta">Inputs: {columnsPreview}</span>
          </span>
        </button>
        <button
          type="button"
          className="canvas-imputer__remove"
          onClick={() => onRemove(operation.id)}
          aria-label={`Remove operation ${index + 1}`}
        >
          Remove
        </button>
      </div>
      {!isCollapsed && (
        <div className="canvas-imputer__card-body" id={bodyId}>
          <div className="canvas-imputer__row">
            <div className="canvas-modal__section-actions">
              <button type="button" className="btn btn-outline-secondary btn-sm" onClick={() => onDuplicate(operation.id)}>
                Duplicate
              </button>
              <button
                type="button"
                className="btn btn-outline-secondary btn-sm"
                onClick={() => onReorder(operation.id, 'up')}
                disabled={index === 0}
              >
                Move up
              </button>
              <button
                type="button"
                className="btn btn-outline-secondary btn-sm"
                onClick={() => onReorder(operation.id, 'down')}
                disabled={index >= total - 1}
              >
                Move down
              </button>
            </div>
          </div>

          {summaryMessage && <p className={summaryNoteClass}>{summaryMessage}</p>}
          {outputColumnsPreview && (
            <p className="canvas-modal__note canvas-modal__note--info">Outputs: {outputColumnsPreview}</p>
          )}

          <div className="canvas-imputer__row">
            <label htmlFor={`feature-math-${operation.id}-type`}>Operation type</label>
            <select
              id={`feature-math-${operation.id}-type`}
              value={operation.type}
              onChange={(event) => onChange(operation.id, { type: event.target.value as FeatureMathOperationType })}
            >
              {FEATURE_MATH_TYPE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="canvas-imputer__row">
            <label htmlFor={`feature-math-${operation.id}-method`}>Method</label>
            <select
              id={`feature-math-${operation.id}-method`}
              value={operation.method}
              onChange={(event) => onChange(operation.id, { method: event.target.value })}
            >
              {methodOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          {operation.type === 'similarity' && similarityDescription && (
            <p className="canvas-modal__note canvas-modal__note--info">{similarityDescription}</p>
          )}

          <div className="canvas-imputer__row">
            <label htmlFor={`feature-math-${operation.id}-input-columns`}>Input columns</label>
            <input
              id={`feature-math-${operation.id}-input-columns`}
              type="text"
              value={inputColumnsValue}
              onChange={(event) => setInputColumnsValue(event.target.value)}
              onBlur={handleInputColumnsCommit}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  handleInputColumnsCommit();
                }
              }}
              placeholder="col_a, col_b"
              list={inputListId}
            />
            <datalist id={inputListId}>
              {availableColumns.map((column) => (
                <option key={`${operation.id}-input-option-${column}`} value={column} />
              ))}
            </datalist>
            {availableColumnsPreview && (
              <p className="canvas-modal__meta">Available: {availableColumnsPreview}</p>
            )}
          </div>

          {showSecondaryField && (
            <div className="canvas-imputer__row">
              <label htmlFor={`feature-math-${operation.id}-secondary-columns`}>
                {operation.type === 'ratio' ? 'Denominator columns' : 'Compare against'}
              </label>
              <input
                id={`feature-math-${operation.id}-secondary-columns`}
                type="text"
                value={secondaryColumnsValue}
                onChange={(event) => setSecondaryColumnsValue(event.target.value)}
                onBlur={handleSecondaryColumnsCommit}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    handleSecondaryColumnsCommit();
                  }
                }}
                placeholder={operation.type === 'ratio' ? 'denominator_col' : 'comparison column'}
                list={secondaryListId}
              />
              <datalist id={secondaryListId}>
                {availableColumns.map((column) => (
                  <option key={`${operation.id}-secondary-option-${column}`} value={column} />
                ))}
              </datalist>
              <p className="canvas-modal__meta">Comma separated list</p>
            </div>
          )}

          {showConstantsField && (
            <div className="canvas-imputer__row">
              <label htmlFor={`feature-math-${operation.id}-constants`}>Constants (optional)</label>
              <input
                id={`feature-math-${operation.id}-constants`}
                type="text"
                value={constantsValue}
                onChange={(event) => setConstantsValue(event.target.value)}
                onBlur={handleConstantsCommit}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    handleConstantsCommit();
                  }
                }}
                placeholder="0.0, 1.5"
              />
              <p className="canvas-modal__meta">Numbers separated by commas are added alongside the selected columns.</p>
            </div>
          )}

          <div className="canvas-imputer__row">
            <label htmlFor={`feature-math-${operation.id}-output-column`}>Output column (optional)</label>
            <input
              id={`feature-math-${operation.id}-output-column`}
              type="text"
              value={operation.outputColumn}
              onChange={(event) => onChange(operation.id, { outputColumn: event.target.value })}
              placeholder="custom_feature_name"
            />
            <p className="canvas-modal__meta">Leave blank to derive a name automatically.</p>
          </div>

          {showPrefixField && (
            <div className="canvas-imputer__row">
              <label htmlFor={`feature-math-${operation.id}-output-prefix`}>Output prefix</label>
              <input
                id={`feature-math-${operation.id}-output-prefix`}
                type="text"
                value={operation.outputPrefix}
                onChange={(event) => onChange(operation.id, { outputPrefix: event.target.value })}
                placeholder="prefix_"
              />
              <p className="canvas-modal__meta">Applied to generated column names (statistics or datetime features).</p>
            </div>
          )}

          {showNumericTuning && (
            <>
              <div className="canvas-imputer__row">
                <label htmlFor={`feature-math-${operation.id}-fillna`}>Fill missing result with</label>
                <input
                  id={`feature-math-${operation.id}-fillna`}
                  type="number"
                  value={operation.fillna ?? ''}
                  onChange={handleNumberChange('fillna')}
                  placeholder="none"
                />
                <p className="canvas-modal__meta">Applies after computing the operation.</p>
              </div>
              <div className="canvas-imputer__row">
                <label htmlFor={`feature-math-${operation.id}-round`}>Round digits</label>
                <input
                  id={`feature-math-${operation.id}-round`}
                  type="number"
                  value={operation.roundDigits ?? ''}
                  onChange={handleNumberChange('roundDigits')}
                  placeholder="none"
                  step={1}
                />
                <p className="canvas-modal__meta">Leave blank for no rounding.</p>
              </div>
            </>
          )}

          {operation.type === 'ratio' && (
            <div className="canvas-imputer__row">
              <label htmlFor={`feature-math-${operation.id}-epsilon`}>Division epsilon</label>
              <input
                id={`feature-math-${operation.id}-epsilon`}
                type="number"
                value={operation.epsilon ?? ''}
                onChange={handleNumberChange('epsilon')}
                placeholder="inherit"
                step="any"
              />
              <p className="canvas-modal__meta">Overrides the node-level epsilon for this ratio.</p>
            </div>
          )}

          {showSimilarityOptions && (
            <div className="canvas-imputer__row">
              <label htmlFor={`feature-math-${operation.id}-normalize`} className="canvas-modal__boolean-control">
                <input
                  id={`feature-math-${operation.id}-normalize`}
                  type="checkbox"
                  checked={operation.normalize}
                  onChange={(event) => onChange(operation.id, { normalize: event.target.checked })}
                />
                <span>Normalize scores to the [0, 1] range</span>
              </label>
              <p className="canvas-modal__meta">Useful when chaining similarity outputs into downstream calculations.</p>
            </div>
          )}

          {showDatetimeFields && (
            <>
              <div className="canvas-imputer__row canvas-imputer__row--wrap">
                <fieldset>
                  <legend>Datetime features</legend>
                  <div className="canvas-modal__checkbox-grid">
                    {DATETIME_FEATURE_OPTIONS.map((option) => {
                      const checkboxId = `feature-math-${operation.id}-datetime-${option.value}`;
                      const checked = operation.datetimeFeatures.includes(option.value);
                      return (
                        <label key={option.value} className="canvas-modal__checkbox-item" htmlFor={checkboxId}>
                          <input
                            id={checkboxId}
                            type="checkbox"
                            checked={checked}
                            onChange={() => handleDateFeatureToggle(option.value)}
                          />
                          <span className="canvas-modal__checkbox-label">{option.label}</span>
                        </label>
                      );
                    })}
                  </div>
                </fieldset>
              </div>
              <div className="canvas-imputer__row">
                <label htmlFor={`feature-math-${operation.id}-timezone`}>Timezone override</label>
                <input
                  id={`feature-math-${operation.id}-timezone`}
                  type="text"
                  value={operation.timezone}
                  onChange={(event) => onChange(operation.id, { timezone: event.target.value })}
                  placeholder="UTC"
                />
                <p className="canvas-modal__meta">Use an IANA timezone identifier, e.g. America/New_York.</p>
              </div>
            </>
          )}

          <div className="canvas-imputer__row">
            <label htmlFor={`feature-math-${operation.id}-allow-overwrite`}>Overwrite behavior</label>
            <select
              id={`feature-math-${operation.id}-allow-overwrite`}
              value={allowOverwriteValue}
              onChange={(event) => {
                const value = event.target.value;
                if (value === 'inherit') {
                  onChange(operation.id, { allowOverwrite: null });
                } else {
                  onChange(operation.id, { allowOverwrite: value === 'allow' });
                }
              }}
            >
              <option value="inherit">Inherit node default</option>
              <option value="allow">Always overwrite existing columns</option>
              <option value="prevent">Never overwrite existing columns</option>
            </select>
          </div>
        </div>
      )}
    </div>
  );
};

type FeatureMathSectionProps = {
  operations: FeatureMathOperationDraft[];
  summaries: FeatureMathOperationSummary[];
  collapsed: ReadonlySet<string>;
  onToggleCollapsed: (operationId: string) => void;
  onAddOperation: (type: FeatureMathOperationType) => void;
  onDuplicateOperation: (operationId: string) => void;
  onRemoveOperation: (operationId: string) => void;
  onReorderOperation: (operationId: string, direction: 'up' | 'down') => void;
  onOperationChange: (operationId: string, updates: Partial<FeatureMathOperationDraft>) => void;
  availableColumns: string[];
  signals: FeatureMathNodeSignal[];
  previewStatus: string;
  errorHandlingParameter: FeatureNodeParameter | null;
  allowOverwriteParameter: FeatureNodeParameter | null;
  defaultTimezoneParameter: FeatureNodeParameter | null;
  epsilonParameter: FeatureNodeParameter | null;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
};

export const FeatureMathSection: React.FC<FeatureMathSectionProps> = ({
  operations,
  summaries,
  collapsed,
  onToggleCollapsed,
  onAddOperation,
  onDuplicateOperation,
  onRemoveOperation,
  onReorderOperation,
  onOperationChange,
  availableColumns,
  signals,
  previewStatus,
  errorHandlingParameter,
  allowOverwriteParameter,
  defaultTimezoneParameter,
  epsilonParameter,
  renderParameterField,
}) => {
  const [pendingType, setPendingType] = useState<FeatureMathOperationType>('arithmetic');
  const summaryMap = useMemo(() => {
    const map = new Map<string, FeatureMathOperationSummary>();
    summaries.forEach((summary) => {
      map.set(summary.id, summary);
    });
    return map;
  }, [summaries]);

  const latestSignal = useMemo(() => {
    if (!signals.length) {
      return null;
    }
    return signals[signals.length - 1];
  }, [signals]);

  const latestWarnings = latestSignal?.warnings ?? [];
  const summaryNoteClass = latestSignal && latestSignal.failed_operations > 0
    ? 'canvas-modal__note canvas-modal__note--warning'
    : 'canvas-modal__note canvas-modal__note--info';

  const handleAddOperation = () => {
    onAddOperation(pendingType);
  };

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Feature math operations</h3>
        <div className="canvas-modal__section-actions">
          <button type="button" className="btn btn-outline-secondary" onClick={handleAddOperation}>
            Add operation
          </button>
        </div>
      </div>

      <p className="canvas-modal__note">
        Blend columns using arithmetic formulas, ratios, statistics, similarity scores, or datetime extractions. Configure
        each operation below — they execute in order.
      </p>

      {previewStatus === 'loading' && <p className="canvas-modal__note">Preview updating…</p>}

      {latestSignal && (
        <p className={summaryNoteClass}>
          Preview applied {latestSignal.applied_operations} of {latestSignal.total_operations} operation
          {latestSignal.total_operations === 1 ? '' : 's'} ({latestSignal.skipped_operations} skipped, {latestSignal.failed_operations} failed).
        </p>
      )}

      {latestWarnings.length > 0 && (
        <ul className="canvas-modal__note-list">
          {latestWarnings.map((warning, index) => (
            <li key={`feature-math-warning-${index}`}>{warning}</li>
          ))}
        </ul>
      )}

      {operations.length === 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          No feature math steps configured yet. Add an operation to create engineered columns.
        </p>
      )}

      {operations.map((operation, index) => (
        <FeatureMathOperationCard
          key={operation.id}
          index={index}
          total={operations.length}
          operation={operation}
          summary={summaryMap.get(operation.id)}
          availableColumns={availableColumns}
          isCollapsed={collapsed.has(operation.id)}
          onToggleCollapsed={onToggleCollapsed}
          onDuplicate={onDuplicateOperation}
          onRemove={onRemoveOperation}
          onReorder={onReorderOperation}
          onChange={onOperationChange}
        />
      ))}

      <div className="canvas-modal__parameter-grid">
        {errorHandlingParameter && renderParameterField(errorHandlingParameter)}
        {allowOverwriteParameter && renderParameterField(allowOverwriteParameter)}
        {defaultTimezoneParameter && renderParameterField(defaultTimezoneParameter)}
        {epsilonParameter && renderParameterField(epsilonParameter)}
      </div>
    </section>
  );
};
