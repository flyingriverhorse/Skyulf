import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Minimize2, Maximize2, X, GripVertical } from 'lucide-react';
import { PERF_FAMILY_DESCRIPTORS } from '../../core/perf/perfThresholds';

interface Props {
  onHide: () => void;
}

interface Position {
  x: number;
  y: number;
}

const STORAGE_KEY = 'skyulf:perf-legend-pos';
const COLLAPSED_KEY = 'skyulf:perf-legend-collapsed';

// Default upper-right corner of the canvas viewport, matching the
// pre-draggable layout so existing users see no jump on upgrade.
const DEFAULT_OFFSET = { right: 12, top: 12 };

function readPos(): Position | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<Position>;
    if (typeof parsed.x === 'number' && typeof parsed.y === 'number') {
      return { x: parsed.x, y: parsed.y };
    }
  } catch {
    /* ignore */
  }
  return null;
}

function writePos(p: Position): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
  } catch {
    /* best-effort */
  }
}

function readCollapsed(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    return window.localStorage.getItem(COLLAPSED_KEY) === '1';
  } catch {
    return false;
  }
}

function writeCollapsed(v: boolean): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(COLLAPSED_KEY, v ? '1' : '0');
  } catch {
    /* best-effort */
  }
}

/**
 * Floating perf-overlay legend.
 *
 * - Draggable from the header (grip icon). Position is persisted to
 *   localStorage so the user's last-placed corner survives reloads.
 * - Collapsible — header-only mode shrinks to a single 1-line pill.
 * - Hide button calls `onHide`, which the parent uses to flip the
 *   global `perfOverlayEnabled` toggle off, keeping the toolbar
 *   button and the legend in sync.
 */
export const PerfOverlayLegend: React.FC<Props> = ({ onHide }) => {
  const [pos, setPos] = useState<Position | null>(() => readPos());
  const [collapsed, setCollapsed] = useState<boolean>(() => readCollapsed());
  const dragRef = useRef<HTMLDivElement | null>(null);
  // Tracks the in-flight drag offset between cursor and card top-left.
  const dragOffsetRef = useRef<{ dx: number; dy: number } | null>(null);

  const onMouseDown = useCallback((e: React.MouseEvent): void => {
    const card = dragRef.current;
    if (!card) return;
    const rect = card.getBoundingClientRect();
    dragOffsetRef.current = { dx: e.clientX - rect.left, dy: e.clientY - rect.top };
    e.preventDefault();
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent): void => {
      const off = dragOffsetRef.current;
      if (!off) return;
      // Clamp to viewport so the user can't drag the card off-screen.
      const card = dragRef.current;
      const w = card?.offsetWidth ?? 220;
      const h = card?.offsetHeight ?? 60;
      const maxX = window.innerWidth - w - 4;
      const maxY = window.innerHeight - h - 4;
      const next: Position = {
        x: Math.max(4, Math.min(maxX, e.clientX - off.dx)),
        y: Math.max(4, Math.min(maxY, e.clientY - off.dy)),
      };
      setPos(next);
    };
    const onUp = (): void => {
      if (dragOffsetRef.current) {
        dragOffsetRef.current = null;
        // Persist the new position only on drop, not on every mousemove.
        setPos((p) => {
          if (p) writePos(p);
          return p;
        });
      }
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, []);

  const toggleCollapsed = (): void => {
    setCollapsed((c) => {
      const next = !c;
      writeCollapsed(next);
      return next;
    });
  };

  const style: React.CSSProperties = pos
    ? { left: pos.x, top: pos.y }
    : { right: DEFAULT_OFFSET.right, top: DEFAULT_OFFSET.top };

  return (
    <div
      ref={dragRef}
      style={style}
      className="pointer-events-auto fixed z-30 rounded-md border border-border bg-background/95 backdrop-blur shadow-md text-[11px] select-none"
    >
      <div
        onMouseDown={onMouseDown}
        role="toolbar"
        aria-label="Perf overlay legend header (drag to move)"
        className="flex items-center gap-1.5 px-2 py-1 cursor-move border-b border-border/60"
      >
        <GripVertical className="w-3 h-3 text-muted-foreground" />
        <span className="font-medium flex-1">Perf overlay</span>
        <button
          type="button"
          onClick={toggleCollapsed}
          className="p-0.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
          aria-label={collapsed ? 'Expand legend' : 'Collapse legend'}
          title={collapsed ? 'Expand' : 'Collapse'}
        >
          {collapsed ? <Maximize2 className="w-3 h-3" /> : <Minimize2 className="w-3 h-3" />}
        </button>
        <button
          type="button"
          onClick={onHide}
          className="p-0.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
          aria-label="Hide performance overlay"
          title="Hide overlay"
        >
          <X className="w-3 h-3" />
        </button>
      </div>
      {!collapsed && (
        <div className="px-2.5 py-2">
          <ul className="space-y-1">
            <li className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-full ring-2 ring-green-500/60 bg-card" />
              <span className="text-muted-foreground">Fast</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-full ring-2 ring-amber-500/70 bg-card" />
              <span className="text-muted-foreground">Medium</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-full ring-2 ring-red-500/70 bg-card" />
              <span className="text-muted-foreground">Slow</span>
            </li>
          </ul>
          <div className="mt-2 space-y-0.5 text-[10px] text-muted-foreground leading-snug">
            {PERF_FAMILY_DESCRIPTORS.map((d) => (
              <div key={d.family}>
                <span className="font-medium text-foreground/80">{d.label}:</span> {d.bandsLabel}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
