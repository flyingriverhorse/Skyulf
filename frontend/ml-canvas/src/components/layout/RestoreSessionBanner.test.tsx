import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { RestoreSessionBanner } from './RestoreSessionBanner';
import { useGraphStore } from '../../core/store/useGraphStore';
import { saveCanvasSnapshot } from '../../core/utils/canvasPersistence';
import { FIT_VIEW_EVENT } from '../../core/hooks/useKeyboardShortcuts';

const LS_KEY = 'skyulf:canvas:autosave:v1';

const sampleNodes = [
  { id: 'a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } },
];

describe('RestoreSessionBanner (CAN-003)', () => {
  beforeEach(() => {
    window.localStorage.clear();
    useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
    vi.restoreAllMocks();
  });

  it('shows nothing when there is no autosave and no unavailable reason', () => {
    render(<RestoreSessionBanner />);
    expect(screen.queryByText(/restore previous session/i)).not.toBeInTheDocument();
  });

  it('labels a valid autosave as "Autosave" with a node count and timestamp', () => {
    saveCanvasSnapshot(sampleNodes, []);
    render(<RestoreSessionBanner />);
    expect(screen.getByText('Autosave')).toBeInTheDocument();
    expect(screen.getByText(/restore previous session/i)).toBeInTheDocument();
    expect(screen.getByText(/1 node/i)).toBeInTheDocument();
  });

  it('restores the snapshot, clears the prompt, and focuses the result on Restore', () => {
    saveCanvasSnapshot(sampleNodes, []);
    const fitViewSpy = vi.fn();
    window.addEventListener(FIT_VIEW_EVENT, fitViewSpy);
    render(<RestoreSessionBanner />);

    fireEvent.click(screen.getByRole('button', { name: 'Restore' }));

    expect(useGraphStore.getState().nodes).toEqual(sampleNodes);
    expect(screen.queryByText(/restore previous session/i)).not.toBeInTheDocument();
    expect(fitViewSpy).toHaveBeenCalledTimes(1);
    window.removeEventListener(FIT_VIEW_EVENT, fitViewSpy);
  });

  it('discards the snapshot and clears storage on Discard', () => {
    saveCanvasSnapshot(sampleNodes, []);
    render(<RestoreSessionBanner />);

    fireEvent.click(screen.getByRole('button', { name: 'Discard' }));

    expect(useGraphStore.getState().nodes).toEqual([]);
    expect(window.localStorage.getItem(LS_KEY)).toBeNull();
    expect(screen.queryByText(/restore previous session/i)).not.toBeInTheDocument();
  });

  it('never overwrites a nonempty canvas — the prompt stays hidden', () => {
    saveCanvasSnapshot(sampleNodes, []);
    useGraphStore.setState({ nodes: sampleNodes, edges: [], executionResult: null });
    render(<RestoreSessionBanner />);
    expect(screen.queryByText(/restore previous session/i)).not.toBeInTheDocument();
  });

  it('explains a corrupt autosave with a non-blocking, dismissible message', () => {
    window.localStorage.setItem(LS_KEY, '{not json');
    render(<RestoreSessionBanner />);

    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.getByText(/corrupted/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Restore' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Dismiss autosave notice'));
    expect(screen.queryByText(/corrupted/i)).not.toBeInTheDocument();
  });

  it('explains a version-mismatched autosave', () => {
    window.localStorage.setItem(
      LS_KEY,
      JSON.stringify({ version: 999, savedAt: new Date().toISOString(), nodes: [], edges: [] }),
    );
    render(<RestoreSessionBanner />);
    expect(screen.getByText(/incompatible version/i)).toBeInTheDocument();
  });

  it('explains a storage read failure (quota exceeded / disabled)', () => {
    vi.spyOn(window.localStorage.__proto__, 'getItem').mockImplementation(() => {
      throw new Error('quota exceeded');
    });
    render(<RestoreSessionBanner />);
    expect(screen.getByText(/full or disabled/i)).toBeInTheDocument();
  });

  it('re-probes after the canvas transitions back to empty (e.g. Clear canvas)', () => {
    saveCanvasSnapshot(sampleNodes, []);
    useGraphStore.setState({ nodes: sampleNodes, edges: [], executionResult: null });
    const { rerender } = render(<RestoreSessionBanner />);
    expect(screen.queryByText(/restore previous session/i)).not.toBeInTheDocument();

    act(() => {
      useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
    });
    rerender(<RestoreSessionBanner />);
    expect(screen.getByText(/restore previous session/i)).toBeInTheDocument();
  });
});
