import React from 'react';
import type { PendingConfigurationDetail } from '../node-settings/utils/pendingConfiguration';

export type PendingConfigurationToastProps = {
  details: PendingConfigurationDetail[];
  onDismiss: () => void;
  onHighlight?: (labels: string[]) => void;
};

const PendingConfigurationToast: React.FC<PendingConfigurationToastProps> = ({
  details,
  onDismiss,
  onHighlight,
}) => {
  if (!details.length) {
    return null;
  }

  const pendingNodes = details.map((detail) => detail.label);
  const primaryLabel = pendingNodes[0];
  const secondaryLabel = pendingNodes[1];
  const remainingCount = pendingNodes.length - 2;
  let subtitle = '';
  if (pendingNodes.length === 1) {
    subtitle = primaryLabel;
  } else if (pendingNodes.length === 2) {
    subtitle = `${primaryLabel} and ${secondaryLabel}`;
  } else if (remainingCount > 0) {
    const plural = remainingCount === 1 ? 'node' : 'nodes';
    subtitle = `${primaryLabel}, ${secondaryLabel} and ${remainingCount} other ${plural}`;
  }

  const firstReason = details.find((detail) => detail.reason)?.reason ?? null;
  const reasonLabel = firstReason
    ? `${firstReason}${details.length > 1 ? '…' : ''}`
    : 'Review highlighted nodes and save their settings.';

  return (
    <div className="pending-toast" role="status" aria-live="polite">
      <div className="pending-toast__icon" aria-hidden="true">
        !
      </div>
      <div className="pending-toast__body">
        <div className="pending-toast__title">Finish node configuration</div>
        <div className="pending-toast__subtitle">
          {subtitle || 'At least one node still needs configuration.'}
        </div>
        {reasonLabel && <div className="pending-toast__hint">{reasonLabel}</div>}
        <div className="pending-toast__actions">
          {onHighlight && (
            <button type="button" className="pending-toast__button" onClick={() => onHighlight(pendingNodes)}>
              Highlight nodes
            </button>
          )}
          <button type="button" className="pending-toast__button pending-toast__button--ghost" onClick={onDismiss}>
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
};

export default PendingConfigurationToast;
