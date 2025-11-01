import React from 'react';
import type { FeatureNodeParameter } from '../../../../api';
import {
  BINNING_DUPLICATE_OPTIONS,
  BINNING_LABEL_FORMAT_OPTIONS,
  BINNING_MISSING_OPTIONS,
  BINNING_STRATEGY_OPTIONS,
  type BinningLabelFormat,
  type BinningMissingStrategy,
  type BinningStrategy,
  type KBinsEncode,
  type KBinsStrategy,
  type NormalizedBinningConfig,
} from './binningSettings';

type BinningFieldIds = {
  equalWidth: string;
  equalFrequency: string;
  precision: string;
  suffix: string;
  missingLabel: string;
  includeLowest: string;
  dropOriginal: string;
  labelFormat: string;
  duplicates: string;
  kbinsNBins: string;
  kbinsEncode: string;
  kbinsStrategy: string;
};

type BinNumericColumnsSectionProps = {
  config: NormalizedBinningConfig;
  columnsParameter: FeatureNodeParameter | null;
  renderMultiSelectField: (parameter: FeatureNodeParameter) => React.ReactNode;
  fieldIds: BinningFieldIds;
  customEdgeDrafts: Record<string, string>;
  customLabelDrafts: Record<string, string>;
  onStrategyChange: (strategy: BinningStrategy) => void;
  onIntegerChange: (
    field: 'equal_width_bins' | 'equal_frequency_bins' | 'precision' | 'kbins_n_bins',
    rawValue: string,
    min: number,
    max: number,
  ) => void;
  onBooleanToggle: (field: 'include_lowest' | 'drop_original', value: boolean) => void;
  onSuffixChange: (value: string) => void;
  onLabelFormatChange: (value: BinningLabelFormat) => void;
  onDuplicatesChange: (value: 'raise' | 'drop') => void;
  onMissingStrategyChange: (value: BinningMissingStrategy) => void;
  onMissingLabelChange: (value: string) => void;
  onCustomBinsChange: (column: string, rawValue: string) => void;
  onCustomLabelsChange: (column: string, rawValue: string) => void;
  onClearCustomColumn: (column: string) => void;
  onKbinsEncodeChange: (value: KBinsEncode) => void;
  onKbinsStrategyChange: (value: KBinsStrategy) => void;
};

export const BinNumericColumnsSection: React.FC<BinNumericColumnsSectionProps> = ({
  config,
  columnsParameter,
  renderMultiSelectField,
  fieldIds,
  customEdgeDrafts,
  customLabelDrafts,
  onStrategyChange,
  onIntegerChange,
  onBooleanToggle,
  onSuffixChange,
  onLabelFormatChange,
  onDuplicatesChange,
  onMissingStrategyChange,
  onMissingLabelChange,
  onCustomBinsChange,
  onCustomLabelsChange,
  onClearCustomColumn,
  onKbinsEncodeChange,
  onKbinsStrategyChange,
}) => {
  const strategyMeta =
    BINNING_STRATEGY_OPTIONS.find((option) => option.value === config.strategy) ?? BINNING_STRATEGY_OPTIONS[0];

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Binning configuration</h3>
      </div>
      <p className="canvas-modal__note">
        Group numeric features into discrete buckets for downstream models or reporting.
      </p>
      <div className="canvas-skewness__segmented" role="tablist" aria-label="Binning strategy">
        {BINNING_STRATEGY_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            className="canvas-skewness__segmented-button"
            data-active={config.strategy === option.value}
            aria-selected={config.strategy === option.value}
            onClick={() => onStrategyChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
      <p className="canvas-modal__note">{strategyMeta.description}</p>

      {columnsParameter && (
        <div className="canvas-modal__subsection">
          <h4>Columns</h4>
          {renderMultiSelectField(columnsParameter)}
        </div>
      )}

      {config.strategy === 'equal_width' && (
        <div className="canvas-modal__subsection">
          <h4>Equal-width bins</h4>
          <div className="canvas-modal__parameter-field">
            <label htmlFor={fieldIds.equalWidth} className="canvas-modal__parameter-label">
              Number of bins
            </label>
            <div className="canvas-modal__parameter-control">
              <input
                id={fieldIds.equalWidth}
                type="number"
                className="canvas-modal__input"
                min={2}
                max={200}
                step={1}
                value={config.equalWidthBins}
                onChange={(event) => onIntegerChange('equal_width_bins', event.target.value, 2, 200)}
              />
            </div>
            <p className="canvas-modal__parameter-description">
              Splits each column’s min/max range into evenly sized intervals.
            </p>
          </div>
        </div>
      )}

      {config.strategy === 'equal_frequency' && (
        <div className="canvas-modal__subsection">
          <h4>Equal-frequency bins</h4>
          <div className="canvas-modal__parameter-grid">
            <div className="canvas-modal__parameter-field">
              <label htmlFor={fieldIds.equalFrequency} className="canvas-modal__parameter-label">
                Number of quantile bins
              </label>
              <div className="canvas-modal__parameter-control">
                <input
                  id={fieldIds.equalFrequency}
                  type="number"
                  className="canvas-modal__input"
                  min={2}
                  max={200}
                  step={1}
                  value={config.equalFrequencyBins}
                  onChange={(event) => onIntegerChange('equal_frequency_bins', event.target.value, 2, 200)}
                />
              </div>
              <p className="canvas-modal__parameter-description">
                Creates buckets with a similar number of rows using quantiles.
              </p>
            </div>
            <div className="canvas-modal__parameter-field">
              <label htmlFor={fieldIds.duplicates} className="canvas-modal__parameter-label">
                Duplicate edges
              </label>
              <div className="canvas-modal__parameter-control">
                <select
                  id={fieldIds.duplicates}
                  value={config.duplicates}
                  onChange={(event) => onDuplicatesChange(event.target.value === 'drop' ? 'drop' : 'raise')}
                >
                  {BINNING_DUPLICATE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              <p className="canvas-modal__parameter-description">
                Control how tied quantile edges are handled during discretization.
              </p>
            </div>
          </div>
        </div>
      )}

      {config.strategy === 'kbins' && (
        <div className="canvas-modal__subsection">
          <h4>KBinsDiscretizer (sklearn)</h4>
          <div className="canvas-modal__parameter-grid">
            <div className="canvas-modal__parameter-field">
              <label htmlFor={fieldIds.kbinsNBins} className="canvas-modal__parameter-label">
                Number of bins
              </label>
              <div className="canvas-modal__parameter-control">
                <input
                  id={fieldIds.kbinsNBins}
                  type="number"
                  className="canvas-modal__input"
                  min={2}
                  max={200}
                  step={1}
                  value={config.kbinsNBins}
                  onChange={(event) => onIntegerChange('kbins_n_bins', event.target.value, 2, 200)}
                />
              </div>
              <p className="canvas-modal__parameter-description">
                Number of bins to produce for each feature.
              </p>
            </div>
            <div className="canvas-modal__parameter-field">
              <label htmlFor={fieldIds.kbinsStrategy} className="canvas-modal__parameter-label">
                Binning strategy
              </label>
              <div className="canvas-modal__parameter-control">
                <select
                  id={fieldIds.kbinsStrategy}
                  value={config.kbinsStrategy}
                  onChange={(event) => onKbinsStrategyChange(event.target.value as KBinsStrategy)}
                >
                  <option value="uniform">Uniform (equal width)</option>
                  <option value="quantile">Quantile (equal frequency)</option>
                  <option value="kmeans">K-means clustering</option>
                </select>
              </div>
              <p className="canvas-modal__parameter-description">
                Strategy for defining bin widths: uniform, quantile, or k-means.
              </p>
            </div>
            <div className="canvas-modal__parameter-field">
              <label htmlFor={fieldIds.kbinsEncode} className="canvas-modal__parameter-label">
                Encoding
              </label>
              <div className="canvas-modal__parameter-control">
                <select
                  id={fieldIds.kbinsEncode}
                  value={config.kbinsEncode}
                  onChange={(event) => onKbinsEncodeChange(event.target.value as KBinsEncode)}
                >
                  <option value="ordinal">Ordinal (0, 1, 2, ...)</option>
                  <option value="onehot">One-hot (sparse)</option>
                  <option value="onehot-dense">One-hot (dense)</option>
                </select>
              </div>
              <p className="canvas-modal__parameter-description">
                How to encode the discretized values.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="canvas-modal__subsection">
        <h4>Output & labels</h4>
        <div className="canvas-modal__parameter-grid">
          <div className="canvas-modal__parameter-field">
            <label htmlFor={fieldIds.suffix} className="canvas-modal__parameter-label">
              Output suffix
            </label>
            <div className="canvas-modal__parameter-control">
              <input
                id={fieldIds.suffix}
                type="text"
                className="canvas-modal__input"
                value={config.outputSuffix}
                onChange={(event) => onSuffixChange(event.target.value)}
                placeholder="e.g. _binned"
              />
            </div>
          </div>
          <div className="canvas-modal__parameter-field">
            <label htmlFor={fieldIds.labelFormat} className="canvas-modal__parameter-label">
              Label format
            </label>
            <div className="canvas-modal__parameter-control">
              <select
                id={fieldIds.labelFormat}
                value={config.labelFormat}
                onChange={(event) => onLabelFormatChange((event.target.value as BinningLabelFormat) ?? 'range')}
              >
                {BINNING_LABEL_FORMAT_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <p className="canvas-modal__parameter-description">
              Controls how new bin values are named when custom labels aren’t provided.
            </p>
          </div>
          <div className="canvas-modal__parameter-field">
            <label htmlFor={fieldIds.precision} className="canvas-modal__parameter-label">
              Decimal precision
            </label>
            <div className="canvas-modal__parameter-control">
              <input
                id={fieldIds.precision}
                type="number"
                className="canvas-modal__input"
                min={0}
                max={8}
                step={1}
                value={config.precision}
                onChange={(event) => onIntegerChange('precision', event.target.value, 0, 8)}
              />
            </div>
            <p className="canvas-modal__parameter-description">Sets rounding precision for interval labels.</p>
          </div>
        </div>
        <div className="canvas-modal__parameter-field">
          <div className="canvas-modal__parameter-label">
            <label htmlFor={fieldIds.includeLowest}>Include lowest edge</label>
          </div>
          <label className="canvas-modal__boolean-control">
            <input
              id={fieldIds.includeLowest}
              type="checkbox"
              checked={config.includeLowest}
              onChange={(event) => onBooleanToggle('include_lowest', event.target.checked)}
            />
            Include the minimum value in the first bin
          </label>
        </div>
        <div className="canvas-modal__parameter-field">
          <div className="canvas-modal__parameter-label">
            <label htmlFor={fieldIds.dropOriginal}>Drop original columns</label>
          </div>
          <label className="canvas-modal__boolean-control">
            <input
              id={fieldIds.dropOriginal}
              type="checkbox"
              checked={config.dropOriginal}
              onChange={(event) => onBooleanToggle('drop_original', event.target.checked)}
            />
            Remove source columns after binning
          </label>
        </div>
      </div>

      <div className="canvas-modal__subsection">
        <h4>Missing values</h4>
        <div className="canvas-skewness__segmented" role="radiogroup" aria-label="Missing value handling">
          {BINNING_MISSING_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              className="canvas-skewness__segmented-button"
              data-active={config.missingStrategy === option.value}
              onClick={() => onMissingStrategyChange(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
        {config.missingStrategy === 'label' && (
          <div className="canvas-modal__parameter-field">
            <label htmlFor={fieldIds.missingLabel} className="canvas-modal__parameter-label">
              Missing label
            </label>
            <div className="canvas-modal__parameter-control">
              <input
                id={fieldIds.missingLabel}
                type="text"
                className="canvas-modal__input"
                value={config.missingLabel}
                onChange={(event) => onMissingLabelChange(event.target.value)}
                placeholder="e.g. Missing"
              />
            </div>
          </div>
        )}
      </div>

      {config.strategy === 'custom' && (
        <div className="canvas-modal__subsection">
          <h4>Custom bin edges</h4>
          <p className="canvas-modal__note">
            Provide ordered numeric thresholds (minimum two) per column. Optional labels override the global label
            format.
          </p>
          {config.columns.length ? (
            <div className="canvas-modal__parameter-list">
              {config.columns.map((column) => {
                const edges = config.customBins[column] ?? [];
                const labels = config.customLabels[column] ?? [];
                const edgeDisplay = Object.prototype.hasOwnProperty.call(customEdgeDrafts, column)
                  ? customEdgeDrafts[column]
                  : edges.length
                    ? edges.join(', ')
                    : '';
                const labelDisplay = Object.prototype.hasOwnProperty.call(customLabelDrafts, column)
                  ? customLabelDrafts[column]
                  : labels.length
                    ? labels.join(', ')
                    : '';
                const hasOverrides = Boolean(edges.length || labels.length);
                return (
                  <div key={`binning-custom-${column}`} className="canvas-modal__parameter-field">
                    <div className="canvas-modal__parameter-label">
                      <span>{column}</span>
                      {hasOverrides && (
                        <div className="canvas-modal__parameter-actions">
                          <button type="button" className="btn btn-link" onClick={() => onClearCustomColumn(column)}>
                            Clear
                          </button>
                        </div>
                      )}
                    </div>
                    <p className="canvas-modal__parameter-description">
                      Bin edges (comma separated, sorted automatically)
                    </p>
                    <div className="canvas-modal__parameter-control">
                      <input
                        type="text"
                        className="canvas-modal__input"
                        value={edgeDisplay}
                        placeholder="e.g. 0, 50, 100"
                        onChange={(event) => onCustomBinsChange(column, event.target.value)}
                      />
                    </div>
                    <p className="canvas-modal__parameter-description">Optional labels (comma separated)</p>
                    <div className="canvas-modal__parameter-control">
                      <input
                        type="text"
                        className="canvas-modal__input"
                        value={labelDisplay}
                        placeholder="e.g. Low, Medium, High"
                        onChange={(event) => onCustomLabelsChange(column, event.target.value)}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="canvas-modal__note">Select at least one column above to define custom thresholds.</p>
          )}
        </div>
      )}
    </section>
  );
};
