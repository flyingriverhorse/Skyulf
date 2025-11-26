import React from 'react';
import { FeatureNodeParameter } from '../../../api';
import { formatMissingPercentage } from '../formatting';

type NodeSettingsParameterFieldProps = {
  parameter: FeatureNodeParameter;
  nodeId: string;
  configState: Record<string, any> | null;
  handleNumberChange: (name: string, value: string) => void;
  handleBooleanChange: (name: string, checked: boolean) => void;
  handleTextChange: (name: string, value: string) => void;
  thresholdParameterName?: string | null;
  normalizedSuggestedThreshold?: number | null;
  showRecommendations?: boolean;
  canApplySuggestedThreshold?: boolean;
  thresholdMatchesSuggestion?: boolean;
  handleApplySuggestedThreshold?: () => void;
  renderMultiSelect: (parameter: FeatureNodeParameter) => React.ReactNode;
};

export const NodeSettingsParameterField: React.FC<NodeSettingsParameterFieldProps> = ({
  parameter,
  nodeId,
  configState,
  handleNumberChange,
  handleBooleanChange,
  handleTextChange,
  thresholdParameterName,
  normalizedSuggestedThreshold,
  showRecommendations,
  canApplySuggestedThreshold,
  thresholdMatchesSuggestion,
  handleApplySuggestedThreshold,
  renderMultiSelect,
}) => {
  if (!parameter?.name) {
    return null;
  }

  if (parameter.type === 'multi_select') {
    return <>{renderMultiSelect(parameter)}</>;
  }

  if (parameter.type === 'number') {
    const inputId = `node-${nodeId}-${parameter.name}`;
    const value = configState?.[parameter.name];
    const numericValue = typeof value === 'number' ? value : value ?? '';

    return (
      <div key={parameter.name} className="canvas-modal__parameter-field">
        <label htmlFor={inputId} className="canvas-modal__parameter-label">
          <span>{parameter.label}</span>
          {parameter.unit && <span className="canvas-modal__parameter-unit">{parameter.unit}</span>}
        </label>
        {parameter.description && (
          <p className="canvas-modal__parameter-description">{parameter.description}</p>
        )}
        <div className="canvas-modal__parameter-control">
          <input
            id={inputId}
            type="number"
            className="canvas-modal__input"
            value={numericValue}
            min={parameter.min !== undefined ? parameter.min : undefined}
            max={parameter.max !== undefined ? parameter.max : undefined}
            step={parameter.step !== undefined ? parameter.step : 'any'}
            onChange={(event) => handleNumberChange(parameter.name, event.target.value)}
          />
          {parameter.unit && (
            <span className="canvas-modal__parameter-unit">{parameter.unit}</span>
          )}
        </div>
        {thresholdParameterName === parameter.name &&
          normalizedSuggestedThreshold !== null &&
          showRecommendations && (
            <div className="canvas-modal__note">
              Suggested threshold from EDA:{' '}
              <strong>{formatMissingPercentage(normalizedSuggestedThreshold)}</strong>
              <button
                type="button"
                className="btn btn-outline-secondary"
                onClick={handleApplySuggestedThreshold}
                disabled={!canApplySuggestedThreshold}
              >
                {thresholdMatchesSuggestion ? 'Applied' : 'Apply suggestion'}
              </button>
            </div>
          )}
      </div>
    );
  }

  if (parameter.type === 'boolean') {
    const inputId = `node-${nodeId}-${parameter.name}`;
    const checked = Boolean(configState?.[parameter.name]);

    return (
      <div key={parameter.name} className="canvas-modal__parameter-field">
        <div className="canvas-modal__parameter-label">
          <label htmlFor={inputId}>{parameter.label}</label>
        </div>
        {parameter.description && (
          <p className="canvas-modal__parameter-description">{parameter.description}</p>
        )}
        <label className="canvas-modal__boolean-control">
          <input
            id={inputId}
            type="checkbox"
            checked={checked}
            onChange={(event) => handleBooleanChange(parameter.name, event.target.checked)}
          />
          <span>{checked ? 'Enabled' : 'Disabled'}</span>
        </label>
      </div>
    );
  }

  if (parameter.type === 'select') {
    const inputId = `node-${nodeId}-${parameter.name}`;
    const options = Array.isArray(parameter.options) ? parameter.options : [];
    const currentValue = configState?.[parameter.name];
    const defaultValue =
      typeof parameter.default === 'string'
        ? parameter.default
        : options.find((option) => option && typeof option.value === 'string')?.value ?? '';
    const value =
      typeof currentValue === 'string' && currentValue.trim().length
        ? currentValue
        : defaultValue;

    return (
      <div key={parameter.name} className="canvas-modal__parameter-field">
        <label htmlFor={inputId} className="canvas-modal__parameter-label">
          {parameter.label}
        </label>
        {parameter.description && (
          <p className="canvas-modal__parameter-description">{parameter.description}</p>
        )}
        <select
          id={inputId}
          className="canvas-modal__input"
          value={value}
          onChange={(event) => handleTextChange(parameter.name, event.target.value)}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value} title={option.description ?? undefined}>
              {option.label ?? option.value}
            </option>
          ))}
        </select>
      </div>
    );
  }

  if (parameter.type === 'textarea') {
    const inputId = `node-${nodeId}-${parameter.name}`;
    
    // Special handling for hyperparameters - convert JSON to simple format
    if (parameter.name === 'hyperparameters') {
      let displayValue = '';
      let internalValue = configState?.[parameter.name];
      
      // Convert JSON to simple key: value format for display
      if (typeof internalValue === 'string' && internalValue.trim()) {
        try {
          const parsed = JSON.parse(internalValue);
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            displayValue = Object.entries(parsed)
              .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
              .join('\n');
          } else {
            displayValue = internalValue;
          }
        } catch {
          displayValue = internalValue;
        }
      } else if (internalValue && typeof internalValue === 'object' && !Array.isArray(internalValue)) {
        displayValue = Object.entries(internalValue)
          .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
          .join('\n');
      }

      return (
        <div key={parameter.name} className="canvas-modal__parameter-field">
          <label htmlFor={inputId} className="canvas-modal__parameter-label">
            {parameter.label}
          </label>
          <p className="canvas-modal__parameter-description">
            {parameter.description || 'Enter parameters as key: value pairs (one per line)'}
          </p>
          <textarea
            id={inputId}
            className="canvas-modal__input canvas-modal__input--wide"
            value={displayValue}
            placeholder={'n_estimators: 100\nmax_depth: 10\nlearning_rate: 0.01'}
            onChange={(event) => {
              const text = event.target.value;
              // Convert simple format to JSON
              const lines = text.split('\n').filter(line => line.trim());
              const params: Record<string, any> = {};
              
              for (const line of lines) {
                const colonIndex = line.indexOf(':');
                if (colonIndex === -1) continue;
                
                const key = line.substring(0, colonIndex).trim();
                const valueStr = line.substring(colonIndex + 1).trim();
                
                if (!key) continue;
                
                // Try to parse the value
                try {
                  params[key] = JSON.parse(valueStr);
                } catch {
                  // If parse fails, treat as string
                  params[key] = valueStr;
                }
              }
              
              // Store as JSON string
              handleTextChange(parameter.name, JSON.stringify(params, null, 2));
            }}
            rows={6}
          />
        </div>
      );
    }

    // Default textarea handling for non-hyperparameters
    const value = typeof configState?.[parameter.name] === 'string' ? configState?.[parameter.name] : '';

    return (
      <div key={parameter.name} className="canvas-modal__parameter-field">
        <label htmlFor={inputId} className="canvas-modal__parameter-label">
          {parameter.label}
        </label>
        {parameter.description && (
          <p className="canvas-modal__parameter-description">{parameter.description}</p>
        )}
        <textarea
          id={inputId}
          className="canvas-modal__input canvas-modal__input--wide"
          value={value}
          placeholder={parameter.placeholder ?? ''}
          onChange={(event) => handleTextChange(parameter.name, event.target.value)}
          rows={4}
        />
      </div>
    );
  }

  const inputId = `node-${nodeId}-${parameter.name}`;
  const value = configState?.[parameter.name] ?? '';

  return (
    <div key={parameter.name} className="canvas-modal__parameter-field">
      <label htmlFor={inputId} className="canvas-modal__parameter-label">
        {parameter.label}
      </label>
      {parameter.description && (
        <p className="canvas-modal__parameter-description">{parameter.description}</p>
      )}
      <input
        id={inputId}
        type="text"
        className="canvas-modal__input canvas-modal__input--wide"
        value={value}
        placeholder={parameter.placeholder ?? ''}
        onChange={(event) => handleTextChange(parameter.name, event.target.value)}
      />
    </div>
  );
};
