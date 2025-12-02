import React from 'react';
import { RotateCcw, Save, Loader2 } from 'lucide-react';

type NodeSettingsFooterProps = {
  onClose: () => void;
  canResetNode: boolean;
  onResetNode: () => void;
  showSaveButton: boolean;
  canSave: boolean;
  onSave: (options?: { closeModal?: boolean }) => void;
  isBusy: boolean;
  busyLabel?: string | null;
  isSaving?: boolean;
  statusMessage?: React.ReactNode;
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
  isSaving = false,
  statusMessage,
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
      {!isBusy && statusMessage}
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
          onClick={() => onSave({ closeModal: false })} 
          disabled={!canSave || isSaving}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: '180px', justifyContent: 'center' }}
        >
          {isSaving ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Saving...
            </>
          ) : (
            <>
              <Save size={16} />
              Save and run changes
            </>
          )}
        </button>
      )}
    </div>
  </div>
);
