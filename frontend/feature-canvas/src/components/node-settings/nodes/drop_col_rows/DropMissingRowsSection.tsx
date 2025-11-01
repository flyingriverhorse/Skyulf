import React from 'react';
import type { FeatureNodeParameter } from '../../../../api';

export type DropMissingRowsSectionProps = {
  thresholdParameter: FeatureNodeParameter | null;
  dropIfAnyParameter: FeatureNodeParameter | null;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
};

export const DropMissingRowsSection: React.FC<DropMissingRowsSectionProps> = ({
  thresholdParameter,
  dropIfAnyParameter,
  renderParameterField,
}) => {
  if (!thresholdParameter && !dropIfAnyParameter) {
    return null;
  }

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Row missingness</h3>
      </div>
      {thresholdParameter && renderParameterField(thresholdParameter)}
      {dropIfAnyParameter && renderParameterField(dropIfAnyParameter)}
    </section>
  );
};
