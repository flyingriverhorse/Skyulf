import React from 'react';
import { Check, AlertTriangle, Circle } from 'lucide-react';
import type { ConnectionInfoSnapshot } from '../hooks';

type ConnectionRequirementsSectionProps = {
  connectionInfo: ConnectionInfoSnapshot;
  connectedInputHandles: Set<string>;
  connectedOutputHandles: Set<string>;
  connectionReady: boolean;
};

export const ConnectionRequirementsSection: React.FC<ConnectionRequirementsSectionProps> = ({
  connectionInfo,
  connectedInputHandles,
  connectedOutputHandles,
  connectionReady,
}) => {
  if (!connectionInfo) {
    return null;
  }

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Connection requirements</h3>
      </div>
      <p
        className={
          connectionReady ? 'canvas-modal__note' : 'canvas-modal__note canvas-modal__note--warning'
        }
        style={{ display: 'flex', alignItems: 'start', gap: '0.5rem' }}
      >
        <span style={{ marginTop: '2px' }}>
          {connectionReady ? <Check size={16} /> : <AlertTriangle size={16} />}
        </span>
        <span>
          <strong>{connectionReady ? 'Ready to execute.' : 'Missing required connections.'}</strong>{' '}
          {connectionReady
            ? 'All required inputs are connected; you can continue configuring this node.'
            : 'Connect the required inputs listed below before running training or evaluation workflows.'}
        </span>
      </p>
      {connectionInfo.inputs && connectionInfo.inputs.length > 0 && (
        <div className="canvas-modal__note">
          <strong>Inputs</strong>
          <ul>
            {connectionInfo.inputs.map((handle) => {
              const isConnected = connectedInputHandles.has(handle.key);
              const isRequired = handle.required !== false;
              return (
                <li key={`inputs-${handle.key}`} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {isConnected ? <Check size={14} /> : isRequired ? <AlertTriangle size={14} /> : <Circle size={14} />}
                  <span>{handle.label}</span>
                  <span>{' - '}</span>
                  <span>{isConnected ? 'Connected' : isRequired ? 'Missing' : 'Optional'}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {connectionInfo.outputs && connectionInfo.outputs.length > 0 && (
        <div className="canvas-modal__note">
          <strong>Outputs</strong>
          <ul>
            {connectionInfo.outputs.map((handle) => {
              const isConnected = connectedOutputHandles.has(handle.key);
              const isRequired = handle.required !== false;
              return (
                <li key={`outputs-${handle.key}`}>
                  <span>{handle.label}</span>
                  <span>{' - '}</span>
                  <span>{isConnected ? 'Connected' : isRequired ? 'Unlinked' : 'Optional'}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
};
