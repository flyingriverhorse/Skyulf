import React, { useMemo } from 'react';
import type { FeatureNodeParameter, PolynomialFeaturesNodeSignal } from '../../../../api';

type PolynomialFeaturesSectionProps = {
  columnsParameter: FeatureNodeParameter | null;
  autoDetectParameter: FeatureNodeParameter | null;
  degreeParameter: FeatureNodeParameter | null;
  includeBiasParameter: FeatureNodeParameter | null;
  interactionOnlyParameter: FeatureNodeParameter | null;
  includeInputFeaturesParameter: FeatureNodeParameter | null;
  outputPrefixParameter: FeatureNodeParameter | null;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  signal: PolynomialFeaturesNodeSignal | null;
};

export const PolynomialFeaturesSection: React.FC<PolynomialFeaturesSectionProps> = ({
  columnsParameter,
  autoDetectParameter,
  degreeParameter,
  includeBiasParameter,
  interactionOnlyParameter,
  includeInputFeaturesParameter,
  outputPrefixParameter,
  renderParameterField,
  signal,
}) => {
  const parameterList = useMemo(() => {
    return [
      columnsParameter,
      autoDetectParameter,
      degreeParameter,
      includeBiasParameter,
      interactionOnlyParameter,
      includeInputFeaturesParameter,
      outputPrefixParameter,
    ].filter((parameter): parameter is FeatureNodeParameter => Boolean(parameter));
  }, [
    columnsParameter,
    autoDetectParameter,
    degreeParameter,
    includeBiasParameter,
    interactionOnlyParameter,
    includeInputFeaturesParameter,
    outputPrefixParameter,
  ]);

  const generatedFeatures = signal?.generated_features ?? [];
  const skippedColumns = signal?.skipped_columns ?? [];
  const filledColumns = signal?.filled_columns ?? {};
  const notes = signal?.notes ?? [];
  const featureCount = signal?.feature_count ?? 0;

  const filledSummary = useMemo(() => {
    const entries = Object.entries(filledColumns);
    if (!entries.length) {
      return null;
    }
    return entries
      .map(([column, count]) => `${column} (${count.toLocaleString()} missing)`)
      .join(', ');
  }, [filledColumns]);

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Polynomial feature settings</h3>
      </div>
      <p className="canvas-modal__note">
        Expand numeric features with polynomial powers and interaction terms to capture non-linear relations. Preview the
        node to inspect generated columns and adjust degree or interaction options as needed.
      </p>
      {parameterList.length > 0 && (
        <div className="canvas-modal__parameter-grid">
          {parameterList.map((parameter) => renderParameterField(parameter))}
        </div>
      )}
      {skippedColumns.length > 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Skipped columns: {skippedColumns.join(', ')}.
        </p>
      )}
      {filledSummary && (
        <p className="canvas-modal__note canvas-modal__note--info">
          Filled missing values with 0 for {filledSummary} before generating features.
        </p>
      )}
      {featureCount > 0 ? (
        <div className="canvas-cast__table-wrapper">
          <table className="canvas-cast__table">
            <thead>
              <tr>
                <th scope="col">Feature column</th>
                <th scope="col">Expression</th>
                <th scope="col">Degree</th>
              </tr>
            </thead>
            <tbody>
              {generatedFeatures.map((feature) => (
                <tr key={`poly-feature-${feature.column}`} className="canvas-cast__row">
                  <th scope="row">{feature.column}</th>
                  <td>{feature.expression || (feature.terms.length ? feature.terms.join(' × ') : '1')}</td>
                  <td>{feature.degree}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="canvas-modal__note">
          No derived features yet. Run a preview to generate polynomial terms for the current configuration.
        </p>
      )}
      {notes.length > 0 && (
        <ul className="canvas-modal__note-list">
          {notes.map((note, index) => (
            <li key={`polynomial-note-${index}`}>{note}</li>
          ))}
        </ul>
      )}
    </section>
  );
};
