import React, { useMemo, useState } from 'react';
import { AlertTriangle, Minimize2 } from 'lucide-react';
import type { PendingConfigurationDetail } from '../node-settings/utils/pendingConfiguration';
import PendingConfigurationToast from './PendingConfigurationToast';

export type PendingConfigurationDockProps = {
  details: PendingConfigurationDetail[];
  onDismiss: () => void;
  onHighlight?: (labels: string[]) => void;
};

const PendingConfigurationDock: React.FC<PendingConfigurationDockProps> = ({
  details,
  onDismiss,
  onHighlight,
}) => {
  const [isCollapsed, setIsCollapsed] = useState(true);
  const visibleCountLabel = useMemo(() => {
    const count = details.length;
    if (count <= 1) {
      return 'Issue in Node';
    }
    return 'Issue in Node';
  }, [details.length]);

  if (!details.length) {
    return null;
  }

  return (
    <div className="pending-dock" data-collapsed={isCollapsed ? 'true' : 'false'}>
      {isCollapsed ? (
        <button
          type="button"
          className="pending-dock__toggle"
          onClick={() => setIsCollapsed(false)}
          aria-label="Expand pending configuration reminder"
        >
          <AlertTriangle size={16} aria-hidden="true" />
          <span>{visibleCountLabel}</span>
        </button>
      ) : (
        <div className="pending-dock__panel">
          <div className="pending-dock__panel-header">
            <span className="pending-dock__panel-label">Pending configuration</span>
            <button
              type="button"
              className="pending-dock__collapse"
              onClick={() => setIsCollapsed(true)}
              aria-label="Minimize reminder"
            >
              <Minimize2 size={16} aria-hidden="true" />
            </button>
          </div>
          <PendingConfigurationToast
            details={details}
            onDismiss={onDismiss}
            onHighlight={(labels) => {
              setIsCollapsed(true);
              onHighlight?.(labels);
            }}
          />
        </div>
      )}
    </div>
  );
};

export default PendingConfigurationDock;
