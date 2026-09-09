import type { ToolbarState } from './_hooks/useToolbarState';
import { Tag, Undo2, Redo2, Keyboard, Command, Trash2 } from 'lucide-react';
import { SHOW_SHORTCUTS_EVENT, SHOW_PALETTE_EVENT } from '../../../core/hooks/useKeyboardShortcuts';
import { ToolbarIconButton } from './ToolbarIconButton';

function ToolbarHistoryControls(
  { editing, layout, readOnly }: Pick<ToolbarState, 'editing' | 'layout' | 'readOnly'>,
) {
  const { undo, redo, canUndo, canRedo } = editing;
  const { hideUndoRedo } = layout;
  return (<>
    {!readOnly && !hideUndoRedo && (
      <ToolbarIconButton
        icon={<Redo2 className="w-4 h-4" />}
        onClick={() => redo()}
        disabled={!canRedo}
        title="Redo (Ctrl+Shift+Z)"
        ariaLabel="Redo"
        testId="toolbar-redo"
      />
    )}
    {!readOnly && !hideUndoRedo && (
      <ToolbarIconButton
        icon={<Undo2 className="w-4 h-4" />}
        onClick={() => undo()}
        disabled={!canUndo}
        title="Undo (Ctrl+Z)"
        ariaLabel="Undo"
        testId="toolbar-undo"
      />
    )}
  </>);
}

export function ToolbarEditingControls(
  { editing, layout, menus, readOnly }: Pick<ToolbarState, 'editing' | 'layout' | 'menus' | 'readOnly'>,
) {
  const { canClear, handleClearCanvas } = editing;
  const { isNarrow } = layout;
  const { showLegend, setShowLegend, legendRef } = menus;
  return (<>
    <div
      ref={legendRef}
      className="flex shrink-0 gap-2"
    >
      {!isNarrow && <>
        <div className="relative">
          <ToolbarIconButton
            icon={<Tag className="w-4 h-4" />}
            onClick={() => setShowLegend((v) => !v)}
            title="Show node badge legend"
            ariaLabel="Show node badge legend"
            ariaExpanded={showLegend}
          />
        </div>
        <ToolbarIconButton
          icon={<Keyboard className="w-4 h-4" />}
          onClick={() => window.dispatchEvent(new CustomEvent(SHOW_SHORTCUTS_EVENT))}
          title="Keyboard shortcuts (?)"
          ariaLabel="Keyboard shortcuts"
        />
        {!readOnly && (
          <ToolbarIconButton
            icon={<Command className="w-4 h-4" />}
            onClick={() => window.dispatchEvent(new CustomEvent(SHOW_PALETTE_EVENT))}
            title="Command palette (Ctrl/Cmd+K)"
            ariaLabel="Open command palette"
          />
        )}
      </>}
      <ToolbarHistoryControls editing={editing} layout={layout} readOnly={readOnly} />
      {!readOnly && !isNarrow && (
        <ToolbarIconButton
          icon={<Trash2 className="w-4 h-4" />}
          onClick={() => { void handleClearCanvas(); }}
          disabled={!canClear}
          title="Clear canvas (Ctrl+Z to undo)"
          ariaLabel="Clear canvas"
          testId="toolbar-clear-canvas"
          variant="danger"
        />
      )}
    </div>
  </>);
}
