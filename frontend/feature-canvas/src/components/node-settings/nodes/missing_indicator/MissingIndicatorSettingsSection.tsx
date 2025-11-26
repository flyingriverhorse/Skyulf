import React from 'react';

type MissingIndicatorSettingsSectionProps = {
  columnsParameter: any;
  suffixParameter: any;
  renderParameterField: (param: any) => React.ReactNode;
};

export const MissingIndicatorSettingsSection: React.FC<MissingIndicatorSettingsSectionProps> = ({
  columnsParameter,
  suffixParameter,
  renderParameterField,
}) => {
  if (!columnsParameter && !suffixParameter) {
    return null;
  }

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Missing indicator settings</h3>
      </div>
      <div className="canvas-modal__parameter-list">
        {columnsParameter ? renderParameterField(columnsParameter) : null}
        {suffixParameter ? renderParameterField(suffixParameter) : null}
      </div>
    </section>
  );
};
