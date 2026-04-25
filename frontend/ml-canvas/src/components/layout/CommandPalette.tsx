import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Search, X } from 'lucide-react';
import { registry } from '../../core/registry/NodeRegistry';
import type { NodeDefinition } from '../../core/types/nodes';
import {
  ADD_NODE_AT_CENTER_EVENT,
  SHOW_PALETTE_EVENT,
  type AddNodeAtCenterDetail,
} from '../../core/hooks/useKeyboardShortcuts';

/**
 * Quick command palette (Ctrl/Cmd+K). Fuzzy filter the registry by
 * label/description/category/type and drop the chosen node at the
 * canvas viewport center. Listens on the global SHOW_PALETTE_EVENT
 * dispatched by `useKeyboardShortcuts` so it can be opened from the
 * keyboard or from any UI affordance without prop-drilling.
 */
export const CommandPalette: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  // All registered node defs are stable for the lifetime of the app
  // (registry is initialised once at boot), so we can compute once.
  const allNodes: NodeDefinition[] = useMemo(() => registry.getAll(), []);

  useEffect(() => {
    const onOpen = (): void => {
      setQuery('');
      setActiveIndex(0);
      setOpen(true);
    };
    window.addEventListener(SHOW_PALETTE_EVENT, onOpen);
    return () => window.removeEventListener(SHOW_PALETTE_EVENT, onOpen);
  }, []);

  // Focus the input on open. requestAnimationFrame so it lands after
  // the modal mounts and the focus-trap-less input becomes selectable.
  useEffect(() => {
    if (!open) return;
    const id = requestAnimationFrame(() => inputRef.current?.focus());
    return () => cancelAnimationFrame(id);
  }, [open]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return allNodes;
    // Cheap fuzzy: substring match against label/description/category/type
    // with a tiny score so exact label-prefix matches surface first.
    const scored = allNodes
      .map((n) => {
        const label = n.label.toLowerCase();
        const desc = n.description.toLowerCase();
        const cat = n.category.toLowerCase();
        const type = n.type.toLowerCase();
        let score = 0;
        if (label.startsWith(q)) score += 100;
        if (label.includes(q)) score += 50;
        if (cat.includes(q)) score += 20;
        if (type.includes(q)) score += 10;
        if (desc.includes(q)) score += 5;
        return { node: n, score };
      })
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score);
    return scored.map((x) => x.node);
  }, [allNodes, query]);

  // Clamp activeIndex when the result set shrinks under the cursor.
  useEffect(() => {
    if (activeIndex >= matches.length) setActiveIndex(0);
  }, [matches.length, activeIndex]);

  const choose = (node: NodeDefinition): void => {
    window.dispatchEvent(
      new CustomEvent<AddNodeAtCenterDetail>(ADD_NODE_AT_CENTER_EVENT, {
        detail: { type: node.type },
      }),
    );
    setOpen(false);
  };

  const onKeyDown = (e: KeyboardEvent | React.KeyboardEvent): void => {
    if (e.key === 'Escape') {
      e.preventDefault();
      setOpen(false);
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, Math.max(matches.length - 1, 0)));
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      const picked = matches[activeIndex];
      if (picked) choose(picked);
    }
  };

  // Window-level keydown so we don't have to bind keyboard handlers on
  // a non-interactive backdrop (eslint jsx-a11y) and so navigation
  // keeps working even if focus drifts off the search input.
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent): void => onKeyDown(e);
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- onKeyDown reads latest state via closures captured each render is fine; we re-bind on every open toggle
  }, [open, matches, activeIndex]);

  // Keep the active row in view as the user arrows through the list.
  useEffect(() => {
    if (!open || !listRef.current) return;
    const el = listRef.current.querySelector<HTMLElement>(
      `[data-palette-index="${activeIndex}"]`,
    );
    el?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex, open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 pt-[15vh]"
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      {/* Backdrop click closes; using a button so a11y rules accept it. */}
      <button
        type="button"
        aria-label="Close palette"
        className="absolute inset-0 cursor-default"
        onClick={() => setOpen(false)}
      />
      <div className="relative w-full max-w-xl mx-4 rounded-xl border border-border bg-background shadow-2xl overflow-hidden">
        <div className="flex items-center gap-2 border-b border-border px-3 py-2">
          <Search className="w-4 h-4 text-muted-foreground" aria-hidden="true" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActiveIndex(0);
            }}
            placeholder="Search nodes by name, category, or description…"
            className="flex-1 bg-transparent outline-none text-sm placeholder:text-muted-foreground"
            aria-label="Search nodes"
          />
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Close"
            className="p-1 rounded hover:bg-accent focus-ring"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
        <ul
          ref={listRef}
          className="max-h-[50vh] overflow-y-auto py-1"
          role="listbox"
          aria-label="Matching nodes"
        >
          {matches.length === 0 && (
            <li className="px-4 py-6 text-center text-sm text-muted-foreground">
              No nodes match &ldquo;{query}&rdquo;
            </li>
          )}
          {matches.map((n, i) => {
            const Icon = n.icon;
            const isActive = i === activeIndex;
            return (
              <li
                key={n.type}
                data-palette-index={i}
                role="option"
                aria-selected={isActive}
              >
                <button
                  type="button"
                  onClick={() => choose(n)}
                  onMouseEnter={() => setActiveIndex(i)}
                  className={`w-full flex items-start gap-3 px-3 py-2 text-left text-sm transition-colors ${
                    isActive ? 'bg-accent' : 'hover:bg-accent/60'
                  }`}
                >
                  {Icon ? (
                    <Icon className="w-4 h-4 mt-0.5 text-primary shrink-0" />
                  ) : (
                    <span className="w-4 h-4 mt-0.5 shrink-0" />
                  )}
                  <span className="flex-1 min-w-0">
                    <span className="flex items-center gap-2">
                      <span className="font-medium truncate">{n.label}</span>
                      <span className="text-[10px] uppercase tracking-wider text-muted-foreground shrink-0">
                        {n.category}
                      </span>
                    </span>
                    <span className="block text-xs text-muted-foreground line-clamp-1">
                      {n.description}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
        <div className="flex items-center justify-between border-t border-border px-3 py-1.5 text-[11px] text-muted-foreground">
          <span>
            <kbd className="px-1 rounded border border-border">↑</kbd>{' '}
            <kbd className="px-1 rounded border border-border">↓</kbd> navigate
          </span>
          <span>
            <kbd className="px-1 rounded border border-border">↵</kbd> insert ·{' '}
            <kbd className="px-1 rounded border border-border">Esc</kbd> close
          </span>
        </div>
      </div>
    </div>
  );
};
