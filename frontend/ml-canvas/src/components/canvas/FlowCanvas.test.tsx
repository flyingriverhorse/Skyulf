import { act, fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { CSSProperties, ReactNode } from 'react';
import { FlowCanvas } from './FlowCanvas';
import { ConfirmProvider } from '../shared/ConfirmDialog';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { FIT_VIEW_EVENT } from '../../core/hooks/useKeyboardShortcuts';

const flow = vi.hoisted(() => ({ fitView: vi.fn(), screenToFlowPosition: vi.fn() }));
vi.mock('@xyflow/react', async importOriginal => ({
  ...await importOriginal<typeof import('@xyflow/react')>(),
  // React Flow owns viewport geometry; the canvas hooks, stores and event subscriptions stay real.
  useReactFlow: () => flow,
  ReactFlow: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  Controls: ({ style }: { style: CSSProperties }) => <div data-testid="zoom-controls" style={style} />,
}));
vi.mock('../../core/api/jobs', async importOriginal => {
  const actual = await importOriginal<typeof import('../../core/api/jobs')>();
  return { ...actual, jobsApi: { ...actual.jobsApi, getNodeSummaries: vi.fn().mockResolvedValue({}) } };
});

describe('FlowCanvas viewport controls', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useGraphStore.setState({ nodes: [], edges: [], executionResult: null, lastRunError: null });
    useViewStore.setState({ readOnlyOverride: 'off', isResultsPanelExpanded: false,
      isResultsPanelMaximized: false, isResultsPanelDismissed: false, perfOverlayEnabled: false, leakageNotice: null });
  });

  it('fits for supported keys, skips editable targets and cleans up keyboard and restore listeners', () => {
    // Viewport shortcuts must preserve native editing and stop acting after the canvas closes.
    const { unmount } = render(<ConfirmProvider><FlowCanvas /><input aria-label="Editor" /></ConfirmProvider>);
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'f' });
    fireEvent.keyDown(window, { key: 'f', altKey: true });
    expect(flow.fitView).not.toHaveBeenCalled();
    fireEvent.keyDown(window, { key: 'F', shiftKey: true });
    fireEvent.keyDown(window, { key: '0', metaKey: true, altKey: true });
    expect(flow.fitView).toHaveBeenCalledTimes(2);
    expect(flow.fitView).toHaveBeenLastCalledWith({ duration: 250, padding: 0.15 });
    act(() => window.dispatchEvent(new Event(FIT_VIEW_EVENT)));
    expect(screen.getByRole('region', { name: 'Pipeline canvas' })).toHaveFocus();
    unmount();
    fireEvent.keyDown(window, { key: 'f' });
    act(() => window.dispatchEvent(new Event(FIT_VIEW_EVENT)));
    expect(flow.fitView).toHaveBeenCalledTimes(3);
  });

  it('lifts controls above visible results and hides them when results are maximized', () => {
    // Zoom buttons must remain reachable as the results panel changes its occupied area.
    render(<ConfirmProvider><FlowCanvas /></ConfirmProvider>);
    const controls = screen.getByTestId('zoom-controls');
    expect(controls).toHaveStyle({ marginBottom: '0px' });
    act(() => useGraphStore.setState({ lastRunError: 'Preview failed' }));
    expect(controls).toHaveStyle({ marginBottom: '40px' });
    act(() => useViewStore.setState({ isResultsPanelExpanded: true }));
    expect(controls.style.marginBottom).toBe('var(--results-panel-height, 384px)');
    act(() => useViewStore.setState({ isResultsPanelMaximized: true }));
    expect(controls).toHaveStyle({ display: 'none' });
    act(() => useViewStore.setState({ isResultsPanelDismissed: true }));
    expect(controls).toHaveStyle({ marginBottom: '0px' });
    expect(controls.style.display).toBe('');
  });
});
