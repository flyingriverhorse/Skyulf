type Direction = 'source' | 'target';
type FromHandle = { nodeId: string; id?: string | null; type: Direction } | null;

/** Normalize forward and reverse drags to the validator's source/target contract. */
export function connectionCandidate(from: FromHandle, direction: Direction, nodeId: string, portId: string) {
  if (!from || from.type === direction) return null;
  return direction === 'target'
    ? { source: from.nodeId, sourceHandle: from.id ?? null, target: nodeId, targetHandle: portId }
    : { source: nodeId, sourceHandle: portId, target: from.nodeId, targetHandle: from.id ?? null };
}

/** Disabled handles keep their own explanation even while a candidate is highlighted. */
export function portAppearance(disabled: boolean, hasCandidate: boolean, issue: string | null, label: string) {
  const guidance = hasCandidate ? issue ? 'incompatible' : 'compatible' : undefined;
  const title = disabled ? 'Validation is disabled. Set Validation Size above zero to include it.'
    : hasCandidate ? issue ?? `Connect to ${label}` : label;
  const color = disabled ? '!bg-muted opacity-40'
    : guidance === 'compatible' ? '!bg-primary ring-2 ring-primary ring-offset-2 ring-offset-card'
      : guidance === 'incompatible' ? '!bg-destructive opacity-60' : '!bg-muted-foreground hover:!bg-primary';
  return { guidance, title, className: `!w-3 !h-3 transition-colors ${color}` };
}
