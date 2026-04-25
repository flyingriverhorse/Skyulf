import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useCanvasAutoSave } from './useCanvasAutoSave';
import { useGraphStore } from '../store/useGraphStore';
import * as persistence from '../utils/canvasPersistence';

describe('useCanvasAutoSave', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('does NOT save on the initial empty graph', () => {
    const save = vi.spyOn(persistence, 'saveCanvasSnapshot').mockImplementation(() => {});
    renderHook(() => useCanvasAutoSave());
    act(() => {
      vi.advanceTimersByTime(2_000);
    });
    expect(save).not.toHaveBeenCalled();
  });

  it('writes to localStorage ~1s after a graph mutation', () => {
    const save = vi.spyOn(persistence, 'saveCanvasSnapshot').mockImplementation(() => {});
    renderHook(() => useCanvasAutoSave());

    act(() => {
      useGraphStore.getState().setGraph(
        [{ id: 'a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } }],
        [],
      );
    });

    // Throttled — nothing yet at 500 ms.
    act(() => vi.advanceTimersByTime(500));
    expect(save).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(600));
    expect(save).toHaveBeenCalledOnce();
  });

  it('coalesces rapid edits onto a single trailing-edge save', () => {
    const save = vi.spyOn(persistence, 'saveCanvasSnapshot').mockImplementation(() => {});
    renderHook(() => useCanvasAutoSave());

    act(() => {
      useGraphStore.getState().setGraph(
        [{ id: 'a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } }],
        [],
      );
    });
    act(() => vi.advanceTimersByTime(300));
    act(() => {
      useGraphStore.getState().setGraph(
        [
          { id: 'a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } },
          { id: 'b', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'encoding' } },
        ],
        [],
      );
    });
    act(() => vi.advanceTimersByTime(300));
    expect(save).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(800));
    expect(save).toHaveBeenCalledOnce();
    // Final saved payload reflects the latest state.
    expect(save.mock.calls[0]?.[0]).toHaveLength(2);
  });

  it('flushes a pending write on unmount when the graph is non-empty', () => {
    const save = vi.spyOn(persistence, 'saveCanvasSnapshot').mockImplementation(() => {});
    const { unmount } = renderHook(() => useCanvasAutoSave());

    act(() => {
      useGraphStore.getState().setGraph(
        [{ id: 'a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } }],
        [],
      );
    });
    // Mid-throttle: timer hasn't fired yet.
    act(() => vi.advanceTimersByTime(200));
    expect(save).not.toHaveBeenCalled();

    unmount();
    expect(save).toHaveBeenCalledOnce();
  });
});
