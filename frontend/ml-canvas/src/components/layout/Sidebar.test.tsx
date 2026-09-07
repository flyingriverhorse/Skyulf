import { fireEvent, render, screen } from '@testing-library/react';
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { initializeRegistry } from '../../core/registry/init';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { FOCUS_NODE_EVENT } from '../../core/hooks/useKeyboardShortcuts';
import { Sidebar } from './Sidebar';

beforeAll(() => { initializeRegistry(); });
beforeEach(() => {
  useGraphStore.setState({ nodes: [], edges: [] });
  useViewStore.setState({ readOnlyOverride: 'off', sidebarOpenOverride: true });
});

describe('component sidebar', () => {
  it('reveals search matches without forgetting subgroup and category choices', () => {
    // A collapsed browsing group must not hide matching nodes or reopen permanently after search.
    render(<Sidebar />);
    const search = screen.getByRole('textbox', { name: 'Search nodes' });
    fireEvent.click(screen.getByRole('button', { name: 'Data cleaning' }));
    expect(screen.getByRole('button', { name: 'Add Imputation node' })).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Preprocessing' }));
    fireEvent.change(search, { target: { value: 'normalize' } });
    expect(screen.getByRole('button', { name: 'Preprocessing' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Add Scaling node' })).toBeVisible();
    fireEvent.change(search, { target: { value: '' } });
    expect(screen.getByRole('button', { name: 'Preprocessing' })).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(screen.getByRole('button', { name: 'Preprocessing' }));
    expect(screen.getByRole('button', { name: 'Data cleaning' })).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('button', { name: 'Numeric & categorical' })).toHaveAttribute('aria-expanded', 'false');
  });

  it('adds and reveals a real node and retains browsing state after reopening', () => {
    // Click insertion must keep the graph and viewport request consistent with the chosen component.
    const listener = vi.fn();
    window.addEventListener(FOCUS_NODE_EVENT, listener);
    try {
      render(<Sidebar />);
      fireEvent.click(screen.getByRole('button', { name: 'Numeric & categorical' }));
      fireEvent.click(screen.getByRole('button', { name: 'Add Scaling node' }));
      const nodes = useGraphStore.getState().nodes;
      expect(nodes).toHaveLength(1);
      expect(nodes[0]?.data.definitionType).toBe('scale_numeric_features');
      expect(listener).toHaveBeenCalledWith(expect.objectContaining({ detail: { id: nodes[0]?.id } }));
      fireEvent.click(screen.getByRole('button', { name: 'Collapse sidebar' }));
      expect(screen.queryByRole('complementary')).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Expand components sidebar' }));
      expect(screen.getByRole('button', { name: 'Numeric & categorical' })).toHaveAttribute('aria-expanded', 'true');
    } finally {
      window.removeEventListener(FOCUS_NODE_EVENT, listener);
    }
  });

  it('provides the registered drag payload and an actionable empty search state', () => {
    // Dragging must preserve node identity even though the library presentation is grouped.
    render(<Sidebar />);
    const transfer = { setData: vi.fn(), effectAllowed: '' };
    fireEvent.dragStart(screen.getByRole('button', { name: 'Add Dataset node' }), { dataTransfer: transfer });
    expect(transfer.setData).toHaveBeenCalledWith('application/reactflow', 'dataset_node');
    expect(transfer.effectAllowed).toBe('move');
    fireEvent.change(screen.getByRole('textbox', { name: 'Search nodes' }), { target: { value: 'no-such-component-xyz' } });
    expect(screen.getByText('No components found')).toBeVisible();
  });
});
