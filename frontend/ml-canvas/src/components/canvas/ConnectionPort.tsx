import { memo } from 'react';
import { Handle, Position, useConnection } from '@xyflow/react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { connectionIssue } from '../../core/utils/connectionValidation';
import type { PortDefinition } from '../../core/types/nodes';
import { ConnectionPicker } from './ConnectionPicker';

/** Highlight compatible ports without subscribing the whole node body to pointer movement. */
export const ConnectionPort = memo(function ConnectionPort({ nodeId, port, direction, top, disabled = false, canConnect = true, compact = false }: {
  nodeId: string; port: PortDefinition; direction: 'source' | 'target'; top: string; disabled?: boolean; canConnect?: boolean; compact?: boolean;
}) {
  const from = useConnection(state => state.fromHandle);
  const nodes = useGraphStore(state => from ? state.nodes : null);
  const edges = useGraphStore(state => from ? state.edges : null);
  const candidate = from && from.type !== direction ? (direction === 'target'
    ? { source: from.nodeId, sourceHandle: from.id ?? null, target: nodeId, targetHandle: port.id }
    : { source: nodeId, sourceHandle: port.id, target: from.nodeId, targetHandle: from.id ?? null }) : null;
  const issue = candidate ? connectionIssue(nodes ?? [], edges ?? [], candidate) : null;
  const guidance = candidate ? issue ? 'incompatible' : 'compatible' : undefined;
  return <Handle id={port.id} type={direction} position={direction === 'target' ? Position.Left : Position.Right}
    isConnectable={canConnect && !disabled}
    aria-label={`${port.label} ${direction === 'target' ? 'input' : 'output'}`}
    data-connection-state={guidance}
    title={disabled ? 'Validation is disabled. Set Validation Size above zero to include it.' : candidate ? issue ?? `Connect to ${port.label}` : port.label}
    className={`!w-3 !h-3 transition-colors ${disabled ? '!bg-muted opacity-40' : guidance === 'compatible' ? '!bg-primary ring-2 ring-primary ring-offset-2 ring-offset-card'
      : guidance === 'incompatible' ? '!bg-destructive opacity-60' : '!bg-muted-foreground hover:!bg-primary'}`}
    style={{ top }}>
    <div data-output-label={direction === 'source' ? port.id : undefined}
      className={`absolute top-1/2 -translate-y-1/2 text-[10px] ${compact ? 'leading-3' : ''} text-muted-foreground whitespace-nowrap rounded bg-card/90 ${direction === 'target' ? 'left-4 px-1 pointer-events-none' : 'right-4'}`}>
      {direction === 'target' || disabled || !canConnect ? port.label : <ConnectionPicker nodeId={nodeId} port={port} compact={compact} />}
    </div>
  </Handle>;
});
