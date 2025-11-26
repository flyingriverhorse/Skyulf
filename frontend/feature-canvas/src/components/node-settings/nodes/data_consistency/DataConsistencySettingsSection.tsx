import React from 'react';

type DataConsistencySettingsSectionProps = {
  parameters: any[];
  renderParameterField: (param: any) => React.ReactNode;
};

export const DataConsistencySettingsSection: React.FC<DataConsistencySettingsSectionProps> = ({
  parameters,
  renderParameterField,
}) => {
  if (parameters.length === 0) {
    return null;
  }

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Data consistency settings</h3>
      </div>
      <div className="canvas-modal__parameter-list">
        {parameters.map((parameter) => renderParameterField(parameter))}
      </div>
    </section>
  );
};
