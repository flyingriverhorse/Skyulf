import { useEffect, useId, useRef, useState } from 'react';
import * as Popover from '@radix-ui/react-popover';
import { Plus, X } from 'lucide-react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { registry } from '../../core/registry/NodeRegistry';
import { searchNodes } from '../../core/utils/nodeSearch';
import { nodeDisplayNames } from '../../core/utils/nodeDisplayNames';
import { connectionIssue } from '../../core/utils/connectionValidation';
import { splitOutputHandles } from '../../core/utils/splitConnections';
import { nextNodePosition } from '../../core/utils/nextNodePosition';
import { getReadOnlyMode, useReadOnlyMode } from '../../core/hooks/useReadOnlyMode';
import { FOCUS_NODE_EVENT, type FocusNodeDetail } from '../../core/hooks/useKeyboardShortcuts';
import type { PortDefinition } from '../../core/types/nodes';

/** Reuse output labels as compact, keyboard-accessible entry points to the next step. */
export function ConnectionPicker({ nodeId, port, compact = false }: { nodeId: string; port: PortDefinition; compact?: boolean }) {
  const [open, setOpen] = useState(false);
  const readOnly = useReadOnlyMode();
  useEffect(() => { if (readOnly) setOpen(false); }, [readOnly]);
  if (readOnly) return <span>{port.label}</span>;
  return <Popover.Root open={open} onOpenChange={setOpen}>
    <Popover.Trigger asChild>
      <button type="button" aria-label={`Next step from ${port.label}`} title="Connect to a node or add the next step"
        className={`nodrag nopan nokey inline-flex items-center gap-1 rounded px-1 ${compact ? 'py-0 leading-3' : 'py-0.5'} hover:bg-accent hover:text-foreground focus-ring`}
        onPointerDown={event => event.stopPropagation()} onMouseDown={event => event.stopPropagation()}
        onTouchStart={event => event.stopPropagation()} onClick={event => event.stopPropagation()}>
        {port.label}<Plus size={10} aria-hidden="true" />
      </button>
    </Popover.Trigger>
    {open && <PickerContent nodeId={nodeId} port={port} close={() => setOpen(false)} />}
  </Popover.Root>;
}

/** Inspect live graph state again when choosing; an open menu must never keep stale endpoints. */
function PickerContent({ nodeId, port, close }: { nodeId: string; port: PortDefinition; close: () => void }) {
  const nodes = useGraphStore(state => state.nodes);
  const edges = useGraphStore(state => state.edges);
  const [mode, setMode] = useState<'new' | 'existing'>('new');
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState('');
  const candidateId = useId();
  const searchRef = useRef<HTMLInputElement>(null);
  const names = nodeDisplayNames(nodes);
  const source = nodes.find(node => node.id === nodeId);
  const definitions = searchNodes(registry.getAll(), query);
  const newChoices = definitions.flatMap(definition => definition.inputs.map(input => ({ definition, input })))
    .filter(({ definition, input }) => !connectionIssue(
      [...nodes, { id: candidateId, position: { x: 0, y: 0 }, data: { definitionType: definition.type } }], edges,
      { source: nodeId, sourceHandle: port.id, target: candidateId, targetHandle: input.id },
    ));
  const existingChoices = nodes.flatMap(node => {
    const definition = registry.get(String(node.data.definitionType));
    if (!definition) return [];
    if (query.trim() && !names.get(node.id)?.toLowerCase().includes(query.trim().toLowerCase()) &&
      searchNodes([{ ...definition, hidden: false }], query).length === 0) return [];
    return definition.inputs.map(input => {
      const connection = { source: nodeId, sourceHandle: port.id, target: node.id, targetHandle: input.id };
      return { node, input, connection, issue: connectionIssue(nodes, edges, connection) };
    });
  }).sort((a, b) => Number(Boolean(a.issue)) - Number(Boolean(b.issue)));

  const add = (type: string, targetHandle: string) => {
    if (!source || getReadOnlyMode()) return;
    const position = nextNodePosition(source, nodes);
    const id = useGraphStore.getState().addConnectedNode(nodeId, port.id, type, targetHandle, position);
    if (!id) { setStatus('No changes made. Choose another step or accept the connection warning.'); return; }
    close();
    window.dispatchEvent(new CustomEvent<FocusNodeDetail>(FOCUS_NODE_EVENT, { detail: { id, relatedNodeIds: [nodeId], focusWrapper: true } }));
  };

  return <Popover.Portal>
    <Popover.Content aria-label="Connect next step" side="right" align="start" sideOffset={12} collisionPadding={12}
      onOpenAutoFocus={event => { event.preventDefault(); searchRef.current?.focus(); }}
      onKeyDown={event => {
        // A background tooltip can consume Radix's document-level Escape before this focused picker.
        if (event.key === 'Escape') { event.stopPropagation(); close(); }
      }}
      className="nodrag nopan nokey z-50 w-80 max-w-[calc(100vw-24px)] overflow-y-auto rounded-lg border bg-popover p-3 text-popover-foreground shadow-lg"
      style={{ maxHeight: 'min(480px, var(--radix-popover-content-available-height))' }}
      onPointerDown={event => event.stopPropagation()} onMouseDown={event => event.stopPropagation()}
      onTouchStart={event => event.stopPropagation()} onClick={event => event.stopPropagation()}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0"><p className="text-sm font-semibold">Next step</p>
          <p className="text-xs text-muted-foreground break-words">{names.get(nodeId)} · {splitOutputHandles(source).length ? splitOutputHandles(source).map(id => registry.get(String(source?.data.definitionType))?.outputs.find(output => output.id === id)?.label).join(' + ') : port.label}</p></div>
        <Popover.Close aria-label="Close next step" className="rounded p-1 focus-ring hover:bg-accent"><X size={14} /></Popover.Close>
      </div>
      <div className="my-3 flex gap-1 rounded-md bg-muted p-1">
        {(['new', 'existing'] as const).map(value => <button key={value} type="button" aria-pressed={mode === value}
          onClick={() => { setMode(value); setStatus(''); }}
          className={`flex-1 rounded px-2 py-1 text-xs focus-ring ${mode === value ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>
          {value === 'new' ? 'Add new node' : 'Existing node'}</button>)}
      </div>
      <input ref={searchRef} aria-label="Search next steps" placeholder="Search nodes or tasks…" value={query}
        onChange={event => setQuery(event.target.value)} className="mb-2 w-full rounded-md border bg-background px-2 py-1.5 text-sm focus-ring" />
      <div className="space-y-1">
        {mode === 'new' ? newChoices.map(({ definition, input }) => <button key={`${definition.type}-${input.id}`} type="button"
          aria-label={`Add ${definition.label} using ${input.label}`} onClick={() => add(definition.type, input.id)}
          className="block w-full rounded-md px-2 py-2 text-left hover:bg-accent focus-ring">
          <span className="block text-sm font-medium">{definition.label} <span className="text-xs text-muted-foreground">· {input.label}</span></span>
          <span className="block text-xs text-muted-foreground">{definition.description}</span>
        </button>) : existingChoices.map(({ node, input, connection, issue }) => <div key={`${node.id}-${input.id}`} className="rounded-md px-2 py-1">
          <button type="button" disabled={Boolean(issue)} aria-label={`Connect to ${names.get(node.id)} using ${input.label}`}
            className="w-full rounded py-1 text-left text-sm enabled:hover:bg-accent disabled:opacity-50 focus-ring"
            onClick={() => {
              if (getReadOnlyMode()) return;
              useGraphStore.getState().onConnect(connection);
              if (useGraphStore.getState().edges.some(edge => edge.source === nodeId && (splitOutputHandles(source).length > 0 || edge.sourceHandle === port.id) && edge.target === node.id && edge.targetHandle === input.id)) close();
              else setStatus('No changes made. The connection warning was cancelled.');
            }}>
            {names.get(node.id)} <span className="text-xs text-muted-foreground">· {input.label}</span>
          </button>
          {issue && <p className="text-xs text-muted-foreground">{issue}</p>}
        </div>)}
        {(mode === 'new' ? newChoices : existingChoices).length === 0 && <p className="p-2 text-xs text-muted-foreground">No matching {mode === 'new' ? 'compatible steps' : 'inputs'}. Try another search or add a node.</p>}
      </div>
      <p role="status" className="mt-2 text-xs text-muted-foreground empty:hidden">{status}</p>
    </Popover.Content>
  </Popover.Portal>;
}
