import * as Popover from '@radix-ui/react-popover';
import { useConnection } from '@xyflow/react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { connectionIssue } from '../../core/utils/connectionValidation';

/** Explain a rejected destination beside the pointer, above nearby nodes and within the viewport. */
export function ConnectionGuidance() {
  const state = useConnection();
  const nodes = useGraphStore(graph => graph.nodes);
  const edges = useGraphStore(graph => graph.edges);
  const from = state.fromHandle;
  const to = state.toHandle;
  if (!state.inProgress || !from || !to) return null;
  const source = from.type === 'source' ? from : to;
  const target = from.type === 'target' ? from : to;
  const issue = from.type === to.type ? 'Choose an input for an output, or an output for an input.' : connectionIssue(nodes, edges, {
    source: source.nodeId, sourceHandle: source.id ?? null, target: target.nodeId, targetHandle: target.id ?? null,
  });
  if (!issue) return null;
  return <Popover.Root open>
    <Popover.Anchor asChild><span aria-hidden="true" className="absolute h-px w-px pointer-events-none"
      style={{ left: state.pointer.x, top: state.pointer.y }} /></Popover.Anchor>
    <Popover.Portal>
      <Popover.Content role="tooltip" aria-label="Connection guidance" side="bottom" sideOffset={16} collisionPadding={12}
        onOpenAutoFocus={event => event.preventDefault()} onCloseAutoFocus={event => event.preventDefault()}
        onInteractOutside={event => event.preventDefault()}
        className="pointer-events-none z-50 max-w-[min(300px,calc(100vw-24px))] rounded-md border border-destructive/40 bg-popover px-3 py-2 text-xs text-popover-foreground shadow-md">
        {issue}
      </Popover.Content>
    </Popover.Portal>
  </Popover.Root>;
}
