import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import * as Popover from '@radix-ui/react-popover';
import { AlertTriangle, X } from 'lucide-react';
import type { CanvasLeakageIssue } from '../../core/types/leakage';

interface LeakageIssuePopoverProps {
  issues: CanvasLeakageIssue[];
  subject: string;
  openGuide: () => void;
  compact?: boolean;
}

/** Persistent, keyboard-accessible feedback shared by canvas nodes and connections. */
export function LeakageIssuePopover({ issues, subject, compact = false }: LeakageIssuePopoverProps) {
  const [open, setOpen] = useState(false);
  const dialogId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const hoverCloseTimer = useRef<ReturnType<typeof setTimeout>>();
  const severity = issues.some(issue => issue.severity === 'error') ? 'error' : 'warning';

  useEffect(() => () => clearTimeout(hoverCloseTimer.current), []);

  const cancelHoverClose = () => clearTimeout(hoverCloseTimer.current);
  const changeOpen = (nextOpen: boolean) => {
    cancelHoverClose();
    setOpen(nextOpen);
  };
  const openDetails = () => changeOpen(true);
  const scheduleHoverClose = () => {
    cancelHoverClose();
    // Allow crossing the gap to the portaled content without dismissing it.
    hoverCloseTimer.current = setTimeout(() => {
      const focused = document.activeElement;
      if (focused === triggerRef.current || contentRef.current?.contains(focused)) return;
      setOpen(false);
    }, 180);
  };

  const close = () => {
    triggerRef.current?.focus({ preventScroll: true });
    changeOpen(false);
  };
  const handleKeyDown = (event: KeyboardEvent) => {
    event.stopPropagation();
    if (event.key === 'Escape') close();
  };

  return <Popover.Root open={open} onOpenChange={changeOpen}>
    <Popover.Anchor asChild>
      <button
        ref={triggerRef}
        type="button"
        aria-label={`Data leakage ${severity}: ${subject}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? dialogId : undefined}
        className={`nodrag nopan flex shrink-0 items-center justify-center rounded-full border shadow-sm focus-ring ${
          compact ? 'h-5 w-5' : 'h-6 w-6'
        } ${severity === 'error'
          ? 'bg-red-50 text-red-600 border-red-300 dark:bg-red-950 dark:text-red-400 dark:border-red-800'
          : 'bg-amber-50 text-amber-600 border-amber-300 dark:bg-amber-950 dark:text-amber-400 dark:border-amber-800'}`}
        onFocus={openDetails}
        onClick={event => { event.stopPropagation(); openDetails(); }}
        onPointerEnter={event => { if (event.pointerType === 'mouse') openDetails(); }}
        onPointerLeave={event => { if (event.pointerType === 'mouse') scheduleHoverClose(); }}
        onPointerDown={event => event.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <AlertTriangle size={compact ? 12 : 14} aria-hidden="true" />
      </button>
    </Popover.Anchor>
    <Popover.Portal>
      <Popover.Content
        ref={contentRef}
        id={dialogId}
        aria-label="Data leakage details"
        side="top"
        align="center"
        sideOffset={8}
        collisionPadding={12}
        className="nodrag nopan z-[60] w-72 max-w-[calc(100vw-24px)] max-h-[calc(100vh-24px)] overflow-y-auto rounded-lg border border-border bg-popover p-3 text-popover-foreground shadow-lg"
        onOpenAutoFocus={event => event.preventDefault()}
        onCloseAutoFocus={event => event.preventDefault()}
        onClick={event => event.stopPropagation()}
        onPointerDown={event => event.stopPropagation()}
        onPointerEnter={cancelHoverClose}
        onPointerLeave={event => { if (event.pointerType === 'mouse') scheduleHoverClose(); }}
        onFocusCapture={cancelHoverClose}
        onKeyDown={handleKeyDown}
      >
        <div className="mb-2 flex items-center justify-between gap-2">
          <p className="text-xs font-semibold">{severity === 'error' ? 'Data leakage risk' : 'Data leakage warning'}</p>
          <button type="button" aria-label="Close data leakage details" onClick={close}
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded text-muted-foreground hover:bg-muted focus-ring">
            <X size={12} aria-hidden="true" />
          </button>
        </div>
        <div className="space-y-3 text-xs">
          {issues.map(issue => <div key={issue.id} className="space-y-1">
            <p>{issue.message}</p>
            <p className="text-muted-foreground">{issue.suggestion}</p>
          </div>)}
        </div>
      </Popover.Content>
    </Popover.Portal>
  </Popover.Root>;
}
