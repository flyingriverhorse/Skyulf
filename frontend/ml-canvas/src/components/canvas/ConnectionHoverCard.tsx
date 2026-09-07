import * as Popover from '@radix-ui/react-popover';

/** Keep endpoint names above canvas nodes and at a readable size at every zoom. */
export function ConnectionHoverCard({ id, x, y, sourceLabel, targetLabel, onEnter, onLeave }: {
  id: string; x: number; y: number; sourceLabel: string; targetLabel: string;
  onEnter: () => void; onLeave: () => void;
}) {
  return <Popover.Root defaultOpen>
    <Popover.Anchor asChild>
      <span aria-hidden="true" className="absolute h-6 w-6 pointer-events-none"
        style={{ transform: `translate(-50%, -50%) translate(${x}px,${y}px)` }} />
    </Popover.Anchor>
    <Popover.Portal>
      <Popover.Content id={id} role="tooltip" aria-label={`${sourceLabel} → ${targetLabel}`}
        side="bottom" sideOffset={10} collisionPadding={12} hideWhenDetached
        updatePositionStrategy="always"
        onOpenAutoFocus={event => event.preventDefault()}
        onCloseAutoFocus={event => event.preventDefault()}
        onInteractOutside={event => event.preventDefault()}
        onPointerDown={event => event.stopPropagation()}
        onClick={event => event.stopPropagation()}
        onMouseEnter={onEnter} onMouseLeave={onLeave}
        className="z-50 w-max max-w-[min(280px,calc(100vw-24px))] overflow-y-auto rounded-md border bg-popover px-2 py-1 text-[11px] leading-snug text-popover-foreground shadow-md [overflow-wrap:anywhere]"
        style={{ maxHeight: 'var(--radix-popover-content-available-height)' }}>
        <span>{sourceLabel}</span>{' → '}<span>{targetLabel}</span>
      </Popover.Content>
    </Popover.Portal>
  </Popover.Root>;
}
