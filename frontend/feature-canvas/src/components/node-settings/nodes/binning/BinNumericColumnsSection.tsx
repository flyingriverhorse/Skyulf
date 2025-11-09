import React from 'react';
import {
  BINNING_LABEL_FORMAT_OPTIONS,
  BINNING_MISSING_OPTIONS,
  type BinningLabelFormat,
  type BinningMissingStrategy,
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
  fieldIds: BinningFieldIds;
  onIntegerChange: (
    field: 'equal_width_bins' | 'equal_frequency_bins' | 'precision' | 'kbins_n_bins',
    rawValue: string,
    min: number,
    max: number,
  ) => void;
  onBooleanToggle: (field: 'include_lowest' | 'drop_original', value: boolean) => void;
  onSuffixChange: (value: string) => void;
  onLabelFormatChange: (value: BinningLabelFormat) => void;
  onMissingStrategyChange: (value: BinningMissingStrategy) => void;
  onMissingLabelChange: (value: string) => void;
};

export const BinNumericColumnsSection: React.FC<BinNumericColumnsSectionProps> = ({
  config,
  fieldIds,
  onIntegerChange,
  onBooleanToggle,
  onSuffixChange,
  onLabelFormatChange,
  onMissingStrategyChange,
  onMissingLabelChange,
}) => {
  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__subsection">
        <h4>Global output & labeling</h4>
        <p className="canvas-modal__note">
          Applies to every selected column unless custom bins or labels override the defaults.
        </p>
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
        <div className="canvas-modal__parameter-grid">
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
    </section>
  );
};
