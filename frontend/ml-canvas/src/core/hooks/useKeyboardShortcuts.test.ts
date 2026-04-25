import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { fireEvent } from '@testing-library/react';
import {
  useKeyboardShortcuts,
  RUN_PREVIEW_EVENT,
  SHOW_SHORTCUTS_EVENT,
} from './useKeyboardShortcuts';
import { useGraphStore } from '../store/useGraphStore';
import { useViewStore } from '../store/useViewStore';

describe('useKeyboardShortcuts', () => {
  beforeEach(() => {
    // Reset stores between cases — the duplicate-action / panel-expanded
    // assertions otherwise leak across tests.
    useGraphStore.setState({ nodes: [], edges: [] });
    useViewStore.getState().setPropertiesPanelExpanded(false);
  });

  it('Ctrl+D triggers duplicateSelectedNodes via the store', () => {
    const dup = vi.spyOn(useGraphStore.getState(), 'duplicateSelectedNodes');
    renderHook(() =>
      useKeyboardShortcuts({ onToggleHelp: () => {}, onCloseHelp: () => {} }),
    );
    fireEvent.keyDown(window, { key: 'd', ctrlKey: true });
    expect(dup).toHaveBeenCalledTimes(1);
  });

  it('Ctrl+Enter dispatches the RUN_PREVIEW_EVENT', () => {
    const listener = vi.fn();
    window.addEventListener(RUN_PREVIEW_EVENT, listener);
    renderHook(() =>
      useKeyboardShortcuts({ onToggleHelp: () => {}, onCloseHelp: () => {} }),
    );
    fireEvent.keyDown(window, { key: 'Enter', ctrlKey: true });
    expect(listener).toHaveBeenCalledTimes(1);
    window.removeEventListener(RUN_PREVIEW_EVENT, listener);
  });

  it('? toggles the help overlay', () => {
    const onToggleHelp = vi.fn();
    renderHook(() =>
      useKeyboardShortcuts({ onToggleHelp, onCloseHelp: () => {} }),
    );
    fireEvent.keyDown(window, { key: '?' });
    expect(onToggleHelp).toHaveBeenCalledTimes(1);
  });

  it('Escape closes help and collapses an expanded properties panel', () => {
    const onCloseHelp = vi.fn();
    useViewStore.getState().setPropertiesPanelExpanded(true);
    renderHook(() =>
      useKeyboardShortcuts({ onToggleHelp: () => {}, onCloseHelp }),
    );
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onCloseHelp).toHaveBeenCalledTimes(1);
    expect(useViewStore.getState().isPropertiesPanelExpanded).toBe(false);
  });

  it('does NOT fire shortcuts when focus is in an input', () => {
    const dup = vi.spyOn(useGraphStore.getState(), 'duplicateSelectedNodes');
    const onToggleHelp = vi.fn();
    renderHook(() =>
      useKeyboardShortcuts({ onToggleHelp, onCloseHelp: () => {} }),
    );

    const input = document.createElement('input');
    document.body.appendChild(input);
    input.focus();

    fireEvent.keyDown(input, { key: 'd', ctrlKey: true, bubbles: true });
    fireEvent.keyDown(input, { key: '?', bubbles: true });
    expect(dup).not.toHaveBeenCalled();
    expect(onToggleHelp).not.toHaveBeenCalled();

    input.remove();
  });

  it('does NOT fire on plain `d` without Ctrl/Cmd', () => {
    const dup = vi.spyOn(useGraphStore.getState(), 'duplicateSelectedNodes');
    renderHook(() =>
      useKeyboardShortcuts({ onToggleHelp: () => {}, onCloseHelp: () => {} }),
    );
    fireEvent.keyDown(window, { key: 'd' });
    expect(dup).not.toHaveBeenCalled();
  });

  it('exports stable custom-event names', () => {
    expect(RUN_PREVIEW_EVENT).toBe('skyulf:run-preview');
    expect(SHOW_SHORTCUTS_EVENT).toBe('skyulf:show-shortcuts');
  });
});
