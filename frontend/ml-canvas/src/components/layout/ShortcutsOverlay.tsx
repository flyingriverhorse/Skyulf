import React, { useEffect } from 'react';
import { X } from 'lucide-react';

interface ShortcutsOverlayProps {
  open: boolean;
  onClose: () => void;
}

interface Shortcut {
  keys: string[];
  description: string;
}

// Keep flat — adding a section header costs more than it pays for at
// this size. If the list grows past ~12 entries, group by "Editing /
// Navigation / Execution" instead of restructuring everything.
const SHORTCUTS: Shortcut[] = [
  { keys: ['Ctrl', 'Z'], description: 'Undo' },
  { keys: ['Ctrl', 'Shift', 'Z'], description: 'Redo' },
  { keys: ['Ctrl', 'D'], description: 'Duplicate selected nodes' },
  { keys: ['Ctrl', 'K'], description: 'Open command palette (insert node)' },
  { keys: ['Delete'], description: 'Delete selected nodes / edges' },
  { keys: ['F'], description: 'Fit view to canvas' },
  { keys: ['Ctrl', '0'], description: 'Fit view to canvas (alt)' },
  { keys: ['Ctrl', 'Enter'], description: 'Run preview' },
  { keys: ['?'], description: 'Toggle this cheatsheet' },
  { keys: ['Esc'], description: 'Close overlays / collapse panels' },
];

const isMac = (): boolean =>
  typeof navigator !== 'undefined' &&
  /Mac|iPhone|iPad|iPod/.test(navigator.platform);

const renderKey = (label: string, mac: boolean): string => {
  if (!mac) return label;
  if (label === 'Ctrl') return '⌘';
  if (label === 'Shift') return '⇧';
  return label;
};

export const ShortcutsOverlay: React.FC<ShortcutsOverlayProps> = ({
  open,
  onClose,
}) => {
  // Esc dismisses the overlay even if the global hook misses it
  // (e.g. focus inside the overlay itself when it has no input).
  useEffect(() => {
    if (!open) return undefined;
    const handler = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open, onClose]);

  if (!open) return null;

  const mac = isMac();

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard shortcuts"
    >
      {/* Backdrop button: click-outside-to-dismiss without violating
          jsx-a11y rules. Esc also closes via the global handler. */}
      <button
        type="button"
        aria-label="Close shortcuts overlay"
        tabIndex={-1}
        onClick={onClose}
        className="absolute inset-0 w-full h-full cursor-default"
      />
      <div
        className="relative w-[26rem] max-w-[92vw] bg-background border rounded-lg shadow-xl p-5"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold">Keyboard shortcuts</h2>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded hover:bg-accent focus-ring"
            aria-label="Close shortcuts overlay"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <ul className="space-y-2 text-sm">
          {SHORTCUTS.map((s) => (
            <li
              key={s.description}
              className="flex items-center justify-between gap-3"
            >
              <span className="text-muted-foreground">{s.description}</span>
              <span className="flex items-center gap-1">
                {s.keys.map((k, i) => (
                  <React.Fragment key={`${s.description}-${k}-${i}`}>
                    {i > 0 && (
                      <span className="text-muted-foreground text-xs">+</span>
                    )}
                    <kbd className="px-1.5 py-0.5 text-xs font-mono bg-muted border rounded">
                      {renderKey(k, mac)}
                    </kbd>
                  </React.Fragment>
                ))}
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-xs text-muted-foreground">
          Shortcuts are disabled while typing in form fields.
        </p>
      </div>
    </div>
  );
};
