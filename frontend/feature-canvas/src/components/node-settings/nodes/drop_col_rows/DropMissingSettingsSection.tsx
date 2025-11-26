import React from 'react';

type DropMissingSettingsSectionProps = {
  thresholdParameter: any;
  dropColumnParameter: any;
  renderParameterField: (param: any) => React.ReactNode;
  renderMultiSelectField: (param: any) => React.ReactNode;
};

export const DropMissingSettingsSection: React.FC<DropMissingSettingsSectionProps> = ({
  thresholdParameter,
  dropColumnParameter,
  renderParameterField,
  renderMultiSelectField,
}) => {
  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Missingness recommendations</h3>
      </div>
      {thresholdParameter && renderParameterField(thresholdParameter)}
      {renderMultiSelectField(dropColumnParameter)}
    </section>
  );
};
