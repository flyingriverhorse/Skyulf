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
  canResetNode?: boolean;
  onResetNode?: () => void;
};

export const NodeSettingsHeader: React.FC<NodeSettingsHeaderProps> = ({
  title,
  isDataset,
  onClose,
  canResetNode = false,
  onResetNode,
}) => (
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
    <div className="canvas-modal__header-actions">
      {canResetNode && onResetNode && (
        <button
          type="button"
          className="btn btn-outline-secondary"
          onClick={onResetNode}
        >
          Reset node
        </button>
      )}
      <button type="button" className="canvas-modal__close" onClick={onClose} aria-label="Close settings">
        ×
      </button>
    </div>
  </div>
);
