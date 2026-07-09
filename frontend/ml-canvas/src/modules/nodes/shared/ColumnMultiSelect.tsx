import { useState, type ReactNode } from 'react';
import { Search } from 'lucide-react';

export interface ColumnMultiSelectProps {
  /** All columns available to pick from. */
  columns: string[];
  /** Currently selected columns. */
  selected: string[];
  onChange: (next: string[]) => void;
  /**
   * Header label, e.g. "Numeric Columns", "Explicitly Drop Columns".
   * Rendered as "{label} (N)" alongside the All/None buttons. Omit to
   * render a plain search+list with no header row (rare — most call
   * sites want the label so the widget is self-describing).
   */
  label?: string;
  /**
   * 'panel' (default): header + All/None buttons. Height behavior is
   * further controlled by `fillHeight` (see below).
   * 'compact': smaller max-height list, no All/None buttons, optional
   * "N selected" footer — used inline within a single-column form
   * alongside other fields (Text Cleaning, Vectorizers, Feature
   * Generation, ...).
   */
  variant?: 'panel' | 'compact';
  /** Only meaningful for variant="compact" — shows "N selected" under the list. */
  showFooterCount?: boolean;
  /** True while the upstream schema is still loading (shows a "Loading columns..." message instead of the list). */
  isLoading?: boolean;
  /** Overrides the default "No columns available" message for the all-empty (no upstream schema) case. */
  emptyMessage?: string;
  /**
   * Optional per-row trailing badge, e.g. ImputationNode's "N missing
   * values filled" count next to each column name. Returning `null`/
   * `undefined` for a given column renders nothing for that row.
   */
  renderItemBadge?: (column: string) => ReactNode;
  /**
   * Single-selection mode: clicking a column replaces the selection with
   * just that column instead of toggling it into a multi-select set. Used
   * where exactly one column is semantically required (e.g. FeatureGeneration's
   * "Column A (Left Operand)" picker for a binary math op). All/None
   * buttons are hidden in this mode (they don't make sense for a single pick).
   */
  single?: boolean;
  /**
   * Only meaningful for variant="panel". Default `true`: the widget fills
   * 100% of its parent's height (`h-full` + an internal `flex-1` list) —
   * for use inside an already height-bounded flex/grid ancestor (e.g. the
   * two-column wide node-settings layouts, where this needs to match its
   * sibling column's height). Set to `false` for panel usage inside a
   * plain stacked/scrolling layout with no bounded ancestor height: the
   * widget then sizes to its own content (capped by `min-h-[160px]` /
   * `max-h-72` with internal scrolling for long lists) instead of forcing
   * a fixed/100% height that leaves empty space under a short list.
   */
  fillHeight?: boolean;
  className?: string;
}

/**
 * Shared column search + multi-select checkbox list.
 *
 * Every preprocessing node needs some form of "pick which columns this
 * applies to" control. Before this component existed, ~19 node settings
 * files each hand-rolled their own version with drifting wording
 * ("All"/"None" vs "Select All"/"Deselect All"), drifting colors
 * (theme tokens vs hardcoded `gray-500`/`blue-500`), and drifting layouts
 * (header count vs footer count). This is the single implementation —
 * please use it instead of writing a new one.
 *
 * "All"/"None" semantics (intentionally unified — the originals disagreed
 * with each other): both act on the *currently filtered/visible* set, not
 * the full column list. "All" merges the visible matches into the existing
 * selection (so a prior search's picks aren't lost); "None" removes only
 * the visible matches, leaving off-screen selections untouched.
 */
export function ColumnMultiSelect({
  columns,
  selected,
  onChange,
  label,
  variant = 'panel',
  showFooterCount = false,
  isLoading = false,
  emptyMessage = 'No columns available',
  single = false,
  renderItemBadge,
  fillHeight = true,
  className = '',
}: ColumnMultiSelectProps) {
  const [searchTerm, setSearchTerm] = useState('');

  const filtered = columns.filter((c) => c.toLowerCase().includes(searchTerm.toLowerCase()));

  const toggle = (col: string) => {
    if (single) {
      onChange([col]);
      return;
    }
    onChange(selected.includes(col) ? selected.filter((c) => c !== col) : [...selected, col]);
  };
  const selectAll = () => { onChange(Array.from(new Set([...selected, ...filtered]))); };
  const selectNone = () => { onChange(selected.filter((c) => !filtered.includes(c))); };

  const isPanel = variant === 'panel';
  const showAllNoneButtons = isPanel && !single;
  const panelFillsHeight = isPanel && fillHeight;

  return (
    <div
      className={`flex flex-col border rounded-md bg-background overflow-hidden ${
        panelFillsHeight ? 'h-full min-h-[200px]' : isPanel ? 'min-h-[160px]' : ''
      } ${className}`}
    >
      <div className={`border-b bg-muted/30 flex flex-col gap-2 shrink-0 ${isPanel ? 'p-2' : 'p-1.5'}`}>
        {(label ?? isPanel) && (
          <div className="flex items-center justify-between">
            {label && (
              <span className="text-xs font-medium text-muted-foreground">
                {label}{!single && ` (${selected.length})`}
              </span>
            )}
            {showAllNoneButtons && (
              <div className={`flex gap-1 ${label ? '' : 'ml-auto'}`}>
                <button
                  type="button"
                  onClick={selectAll}
                  className="text-[10px] px-2 py-1 hover:bg-accent rounded text-muted-foreground hover:text-foreground transition-colors"
                >
                  All
                </button>
                <button
                  type="button"
                  onClick={selectNone}
                  className="text-[10px] px-2 py-1 hover:bg-accent rounded text-muted-foreground hover:text-foreground transition-colors"
                >
                  None
                </button>
              </div>
            )}
          </div>
        )}
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground pointer-events-none" />
          <input
            type="text"
            placeholder="Search columns..."
            value={searchTerm}
            onChange={(e) => { setSearchTerm(e.target.value); }}
            className="w-full text-xs pl-6 pr-2 py-1.5 border rounded bg-background outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-all"
          />
        </div>
      </div>

      <div className={`overflow-y-auto p-1 space-y-0.5 ${panelFillsHeight ? 'flex-1' : isPanel ? 'max-h-72' : 'max-h-40'}`}>
        {isLoading ? (
          <div className="text-xs text-muted-foreground italic p-8 text-center">Loading columns...</div>
        ) : columns.length === 0 ? (
          <div className="text-xs text-muted-foreground italic p-8 text-center">{emptyMessage}</div>
        ) : filtered.length === 0 ? (
          <div className="text-xs text-muted-foreground text-center py-8">
            No columns match &quot;{searchTerm}&quot;
          </div>
        ) : (
          filtered.map((col) => {
            const isSelected = selected.includes(col);
            const badge = renderItemBadge?.(col);
            return (
              <label
                key={col}
                className="flex items-center justify-between gap-2 text-xs px-2 py-1.5 rounded cursor-pointer select-none transition-colors hover:bg-accent/50"
              >
                <span className="flex items-center gap-2 overflow-hidden">
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => { toggle(col); }}
                    className="rounded border-gray-300 text-primary focus:ring-primary w-3.5 h-3.5 shrink-0"
                  />
                  <span className="truncate font-mono" title={col}>{col}</span>
                </span>
                {badge}
              </label>
            );
          })
        )}
      </div>

      {!isPanel && showFooterCount && (
        <div className="text-[10px] text-muted-foreground text-right px-2 py-1 border-t shrink-0">
          {selected.length} selected
        </div>
      )}
    </div>
  );
}
