import type { MutableRefObject } from 'react';

interface PanelResizeHandleProps {
  panelId: string;
  panelHeight: number;
  maxHeight: number;
  dragStart: MutableRefObject<{ y: number; height: number } | null>;
  resizeTo: (height: number) => void;
  setResultsPanelHeight: (height: number) => void;
  stopResizing: () => void;
}

/** Render the pointer and keyboard handle without owning the drag lifetime. */
export function PanelResizeHandle({
  panelId,
  panelHeight,
  maxHeight,
  dragStart,
  resizeTo,
  setResultsPanelHeight,
  stopResizing,
}: PanelResizeHandleProps) {
  return (
    // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex -- A focusable ARIA separator is an interactive pane-resize widget.
    <div role="separator" tabIndex={0}
      aria-label="Resize results panel"
      aria-orientation="horizontal"
      aria-controls={panelId}
      aria-valuemin={200}
      aria-valuemax={maxHeight}
      aria-valuenow={panelHeight}
      aria-valuetext={`${panelHeight} pixels tall`}
      title="Drag to resize. Up arrow grows; Down arrow shrinks. Home resets; End maximizes."
      className="absolute inset-x-0 -top-1 z-20 h-2 cursor-row-resize touch-none hover:bg-primary/20 focus-visible:bg-primary/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary"
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.currentTarget.focus();
        event.currentTarget.setPointerCapture(event.pointerId);
        dragStart.current = { y: event.clientY, height: panelHeight };
      }}
      onPointerMove={(event) => {
        if (dragStart.current) resizeTo(dragStart.current.height + dragStart.current.y - event.clientY);
      }}
      onPointerUp={stopResizing}
      onPointerCancel={stopResizing}
      onLostPointerCapture={stopResizing}
      onKeyDown={(event) => {
        const heights: Record<string, number> = {
          ArrowUp: panelHeight + 20,
          ArrowDown: panelHeight - 20,
          Home: 384,
          End: maxHeight,
        };
        const height = heights[event.key];
        if (height === undefined) return;
        event.preventDefault();
        // Home resets the preference even when this viewport cannot fit it.
        if (event.key === 'Home') setResultsPanelHeight(height);
        else resizeTo(height);
      }}
    >
      <span aria-hidden="true" className="absolute left-1/2 top-1/2 h-1 w-8 -translate-x-1/2 -translate-y-1/2 rounded-full bg-muted-foreground/40" />
    </div>
  );
}
