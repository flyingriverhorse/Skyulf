import React from 'react';
import { RotateCcw, Save } from 'lucide-react';

type NodeSettingsFooterProps = {
  onClose: () => void;
  canResetNode: boolean;
  onResetNode: () => void;
  showSaveButton: boolean;
  canSave: boolean;
  onSave: () => void;
  isBusy: boolean;
  busyLabel?: string | null;
};

export const NodeSettingsFooter: React.FC<NodeSettingsFooterProps> = ({
  onClose,
  canResetNode,
  onResetNode,
  showSaveButton,
  canSave,
  onSave,
  isBusy,
  busyLabel,
}) => (
  <div className="canvas-modal__footer">
    <div
      className="canvas-modal__footer-status"
      role="status"
      aria-live="polite"
      aria-busy={isBusy}
    >
      {isBusy && (
        <>
          <span className="canvas-modal__footer-spinner" aria-hidden="true" />
          <span>{busyLabel ?? 'Processing…'}</span>
        </>
      )}
    </div>
    <div className="canvas-modal__footer-actions">
      <button type="button" className="btn btn-outline-secondary" onClick={onClose}>
        Close
      </button>
      {canResetNode && (
        <button
          type="button"
          className="btn btn-outline-secondary canvas-modal__reset"
          onClick={onResetNode}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <RotateCcw size={16} />
          Reset node
        </button>
      )}
      {showSaveButton && (
        <button 
          type="button" 
          className="btn btn-primary" 
          onClick={onSave} 
          disabled={!canSave}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <Save size={16} />
          Save changes
        </button>
      )}
    </div>
  </div>
);
