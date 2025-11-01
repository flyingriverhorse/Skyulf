import React from 'react';

type PlaceholderIconProps = {
  type: 'dataset' | 'default';
};

const PlaceholderIcon: React.FC<PlaceholderIconProps> = ({ type }) => (
  <span
    aria-hidden="true"
    className={`canvas-modal__title-glyph${type === 'dataset' ? ' canvas-modal__title-glyph--dataset' : ''}`}
  >
    {type === 'dataset' ? '📊' : '⚙️'}
  </span>
);

type NodeSettingsHeaderProps = {
  title: string;
  isDataset: boolean;
  onClose: () => void;
};

export const NodeSettingsHeader: React.FC<NodeSettingsHeaderProps> = ({ title, isDataset, onClose }) => (
  <div className="canvas-modal__header">
    <div className="canvas-modal__title-group">
      <PlaceholderIcon type={isDataset ? 'dataset' : 'default'} />
      <div>
        <h2 className="canvas-modal__title" id="node-settings-title">
          {title}
        </h2>
        <p className="canvas-modal__subtitle">
          {isDataset
            ? 'Configure how this dataset feeds the pipeline.'
            : 'Adjust parameters to control this transformation.'}
        </p>
      </div>
    </div>
    <button type="button" className="canvas-modal__close" onClick={onClose} aria-label="Close settings">
      ×
    </button>
  </div>
);
