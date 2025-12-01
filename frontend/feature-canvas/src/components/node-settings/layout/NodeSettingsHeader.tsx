import React from 'react';
import { Database, Settings, X, RotateCcw } from 'lucide-react';

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
      <span
        className={`canvas-modal__title-glyph${isDataset ? ' canvas-modal__title-glyph--dataset' : ''}`}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
      >
        {isDataset ? <Database size={24} strokeWidth={1.5} /> : <Settings size={24} strokeWidth={1.5} />}
      </span>
      <div>
        <h2 className="canvas-modal__title" id="node-settings-title">
          {title}
        </h2>
      </div>
    </div>
    <div className="canvas-modal__header-actions">
      {canResetNode && onResetNode && (
        <button
          type="button"
          className="btn btn-outline-secondary"
          onClick={onResetNode}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <RotateCcw size={16} />
          Reset node
        </button>
      )}
      <button 
        type="button" 
        className="canvas-modal__close" 
        onClick={onClose} 
        aria-label="Close settings"
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
      >
        <X size={24} strokeWidth={1.5} />
      </button>
    </div>
  </div>
);
