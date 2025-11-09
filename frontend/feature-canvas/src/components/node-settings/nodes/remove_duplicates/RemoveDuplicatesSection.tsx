import React from 'react';
import type { FeatureNodeParameter } from '../../../../api';
import { DEDUP_KEEP_OPTIONS, type KeepStrategy } from './removeDuplicatesSettings';

export type RemoveDuplicatesSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  removeDuplicatesColumnsParameter: FeatureNodeParameter | null;
  removeDuplicatesKeepParameter: FeatureNodeParameter | null;
  renderMultiSelectField: (parameter: FeatureNodeParameter) => React.ReactNode;
  removeDuplicatesKeepSelectId: string;
  removeDuplicatesKeep: KeepStrategy;
  onKeepChange: (value: KeepStrategy) => void;
};

export const RemoveDuplicatesSection: React.FC<RemoveDuplicatesSectionProps> = ({
  sourceId,
  hasReachableSource,
  removeDuplicatesColumnsParameter,
  removeDuplicatesKeepParameter,
  renderMultiSelectField,
  removeDuplicatesKeepSelectId,
  removeDuplicatesKeep,
  onKeepChange,
}) => {
  if (!removeDuplicatesColumnsParameter && !removeDuplicatesKeepParameter) {
    return null;
  }

  const handleSelectChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const rawValue = event.target.value as KeepStrategy;
    const normalized: KeepStrategy = rawValue === 'last' || rawValue === 'none' ? rawValue : 'first';
    onKeepChange(normalized);
  };

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Duplicate handling</h3>
      </div>
      <p className="canvas-modal__note">
        Choose comparison columns and decide whether to keep the first row, last row, or drop all duplicates.
      </p>
      {!sourceId ? (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Select a dataset to load the available columns for duplicate checks.
        </p>
      ) : !hasReachableSource ? (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Connect this node to an upstream output to inspect its columns.
        </p>
      ) : null}
      {removeDuplicatesColumnsParameter && (
        <div className="canvas-modal__subsection">
          <h4>Columns to compare</h4>
          {renderMultiSelectField(removeDuplicatesColumnsParameter)}
        </div>
      )}
      {removeDuplicatesKeepParameter && (
        <div className="canvas-modal__parameter-field">
          <label htmlFor={removeDuplicatesKeepSelectId} className="canvas-modal__parameter-label">
            {removeDuplicatesKeepParameter.label ?? 'Keep strategy'}
          </label>
          {removeDuplicatesKeepParameter.description && (
            <p className="canvas-modal__parameter-description">
              {removeDuplicatesKeepParameter.description}
            </p>
          )}
          <div className="canvas-modal__parameter-control">
            <select
              id={removeDuplicatesKeepSelectId}
              value={removeDuplicatesKeep}
              onChange={handleSelectChange}
            >
              {DEDUP_KEEP_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </section>
  );
};
