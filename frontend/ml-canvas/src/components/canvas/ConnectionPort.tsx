import { memo } from 'react';
import { Handle, Position, useConnection } from '@xyflow/react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { connectionIssue } from '../../core/utils/connectionValidation';
import type { PortDefinition } from '../../core/types/nodes';
import { ConnectionPicker } from './ConnectionPicker';
import { connectionCandidate, portAppearance } from './connectionPort/portPresentation';

/** Highlight compatible ports without subscribing the whole node body to pointer movement. */
export const ConnectionPort = memo(function ConnectionPort({ nodeId, port, direction, top, disabled = false, canConnect = true, compact = false }: {
  nodeId: string; port: PortDefinition; direction: 'source' | 'target'; top: string; disabled?: boolean; canConnect?: boolean; compact?: boolean;
}) {
  const from = useConnection(state => state.fromHandle);
  const nodes = useGraphStore(state => from ? state.nodes : null);
  const edges = useGraphStore(state => from ? state.edges : null);
  const candidate = connectionCandidate(from, direction, nodeId, port.id);
  const issue = candidate ? connectionIssue(nodes ?? [], edges ?? [], candidate) : null;
  const { guidance, title, className } = portAppearance(disabled, candidate !== null, issue, port.label);
  return <Handle id={port.id} type={direction} position={direction === 'target' ? Position.Left : Position.Right}
    isConnectable={canConnect && !disabled}
    aria-label={`${port.label} ${direction === 'target' ? 'input' : 'output'}`}
    data-connection-state={guidance}
    title={title}
    className={className}
    style={{ top }}>
    <PortLabel nodeId={nodeId} port={port} direction={direction} compact={compact} disabled={disabled} canConnect={canConnect} />
  </Handle>;
});

function PortLabel({ nodeId, port, direction, compact, disabled, canConnect }: {
  nodeId: string; port: PortDefinition; direction: 'source' | 'target'; compact: boolean; disabled: boolean; canConnect: boolean;
}) {
  return (
    <div data-output-label={direction === 'source' ? port.id : undefined}
      className={`absolute top-1/2 -translate-y-1/2 text-[10px] ${compact ? 'leading-3' : ''} text-muted-foreground whitespace-nowrap rounded bg-card/90 ${direction === 'target' ? 'left-4 px-1 pointer-events-none' : 'right-4'}`}>
      {direction === 'target' || disabled || !canConnect ? port.label : <ConnectionPicker nodeId={nodeId} port={port} compact={compact} />}
    </div>
  );
}
