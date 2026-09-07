import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { initializeRegistry } from '../../core/registry/init';
import { registry } from '../../core/registry/NodeRegistry';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { FOCUS_NODE_EVENT } from '../../core/hooks/useKeyboardShortcuts';
import { ConnectionPicker } from './ConnectionPicker';

beforeAll(() => { initializeRegistry(); });
beforeEach(() => {
  useGraphStore.setState({ nodes: [], edges: [] });
  useGraphStore.temporal.getState().clear();
  useViewStore.setState({ readOnlyOverride: 'off' });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useViewStore.setState({ readOnlyOverride: 'auto' });
});

/** Open an actual registered output so compatibility uses the production port contract. */
function openPicker(type: string, handle: string) {
  const source = useGraphStore.getState().addNode(type, { x: 0, y: 0 });
  const port = registry.get(type)?.outputs.find(output => output.id === handle);
  if (!port) throw new Error(`Missing output ${type}.${handle}`);
  render(<ConnectionPicker nodeId={source} port={port} />);
  fireEvent.click(screen.getByRole('button', { name: `Next step from ${port.label}` }));
  return source;
}

describe('ConnectionPicker', () => {
  it('focuses the search and offers only an ensemble after a trained model', async () => {
    // Keyboard users must find valid next steps without being offered data-processing inputs.
    openPicker('classification', 'model');
    const search = screen.getByRole('textbox', { name: 'Search next steps' });
    await waitFor(() => expect(search).toHaveFocus());
    expect(screen.getAllByRole('button', { name: /^Add .* using / })).toHaveLength(1);
    expect(screen.getByRole('button', { name: 'Add Ensemble using Data / Models' })).toBeEnabled();

    fireEvent.change(search, { target: { value: 'imputation' } });
    expect(screen.queryByRole('button', { name: /^Add .* using / })).not.toBeInTheDocument();
    expect(screen.getByText(/No matching compatible steps/)).toBeInTheDocument();
    fireEvent.change(search, { target: { value: 'ensemble' } });
    expect(screen.getAllByRole('button', { name: /^Add .* using / })).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: 'Close next step' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(useGraphStore.getState().edges).toHaveLength(0);
  });

  it('adds a selected node, connects the split set, and requests focus after closing', () => {
    // Choosing Test must preserve the grouped split connection and reveal the new node.
    const source = openPicker('TrainTestSplitter', 'test');
    const dispatch = vi.spyOn(window, 'dispatchEvent');
    fireEvent.change(screen.getByRole('textbox', { name: 'Search next steps' }), {
      target: { value: 'imputation' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add Imputation using Data' }));

    const { nodes, edges } = useGraphStore.getState();
    const added = nodes.find(node => node.id !== source);
    expect(nodes).toHaveLength(2);
    expect(added).toMatchObject({ selected: true, data: { definitionType: 'imputation_node' } });
    expect(added?.position.x).toBeGreaterThan(0);
    expect(edges).toEqual([expect.objectContaining({
      source, sourceHandle: 'train', target: added?.id, targetHandle: 'in',
    })]);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(dispatch).toHaveBeenCalledWith(expect.objectContaining({
      type: FOCUS_NODE_EVENT,
      detail: { id: added?.id, relatedNodeIds: [source], focusWrapper: true },
    }));
  });

  it('finds existing nodes by their labels and disables duplicate connections', () => {
    // Existing steps should be reusable without adding another node or duplicate wire.
    const target = useGraphStore.getState().addNode('imputation_node', { x: 350, y: 0 }, { label: 'Clean customers' });
    const source = openPicker('TrainTestSplitter', 'train');
    fireEvent.click(screen.getByRole('button', { name: 'Existing node' }));
    expect(screen.getByRole('button', { name: 'Existing node' })).toHaveAttribute('aria-pressed', 'true');
    fireEvent.change(screen.getByRole('textbox', { name: 'Search next steps' }), {
      target: { value: 'customers' },
    });
    const connectLabel = 'Connect to Clean customers using Data';
    expect(screen.getAllByRole('button', { name: /^Connect to / })).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: connectLabel }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(useGraphStore.getState().edges).toEqual([expect.objectContaining({
      source, target, sourceHandle: 'train', targetHandle: 'in',
    })]);

    fireEvent.click(screen.getByRole('button', { name: 'Next step from Train' }));
    fireEvent.click(screen.getByRole('button', { name: 'Existing node' }));
    expect(screen.getByRole('button', { name: connectLabel })).toBeDisabled();
    expect(screen.getByText(/These ports are already connected/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: connectLabel }));
    expect(useGraphStore.getState().nodes).toHaveLength(2);
    expect(useGraphStore.getState().edges).toHaveLength(1);
  });

  it('keeps both insertion and existing-node choices unchanged when a warning is cancelled', () => {
    // Declining the leakage warning must leave no orphan node, edge, or undo entry.
    useGraphStore.getState().addNode('imputation_node', { x: 350, y: 0 });
    openPicker('feature_target_split', 'X');
    useGraphStore.temporal.getState().clear();
    const originalNodes = useGraphStore.getState().nodes;
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    fireEvent.change(screen.getByRole('textbox', { name: 'Search next steps' }), {
      target: { value: 'imputation' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add Imputation using Data' }));
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('possible data leakage'));
    expect(screen.getByRole('status')).toHaveTextContent('No changes made. Choose another step');
    expect(useGraphStore.getState().nodes).toEqual(originalNodes);

    fireEvent.click(screen.getByRole('button', { name: 'Existing node' }));
    expect(screen.getByRole('status')).toBeEmptyDOMElement();
    fireEvent.click(screen.getByRole('button', { name: 'Connect to Imputation using Data' }));
    expect(confirm).toHaveBeenCalledTimes(2);
    expect(screen.getByRole('status')).toHaveTextContent('No changes made. The connection warning was cancelled.');
    expect(screen.getByRole('dialog', { name: 'Connect next step' })).toBeInTheDocument();
    expect(useGraphStore.getState().nodes).toEqual(originalNodes);
    expect(useGraphStore.getState().edges).toHaveLength(0);
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(0);
  });

  it('removes stale choices when the source disappears while the picker is open', () => {
    // Deleting the source must not leave an actionable menu that creates disconnected nodes.
    openPicker('TrainTestSplitter', 'train');
    expect(screen.getByRole('button', { name: 'Add Imputation using Data' })).toBeEnabled();
    act(() => { useGraphStore.setState({ nodes: [], edges: [] }); });
    expect(screen.queryByRole('button', { name: /^Add .* using / })).not.toBeInTheDocument();
    expect(screen.getByText(/No matching compatible steps/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close next step' }));
    expect(useGraphStore.getState().nodes).toHaveLength(0);
    expect(useGraphStore.getState().edges).toHaveLength(0);
  });

  it('closes the menu and leaves only the output label after switching to read-only', () => {
    // A mode change must immediately withdraw editing controls without changing the graph.
    openPicker('TrainTestSplitter', 'train');
    const originalNodes = useGraphStore.getState().nodes;
    act(() => { useViewStore.setState({ readOnlyOverride: 'on' }); });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Next step from Train' })).not.toBeInTheDocument();
    expect(screen.getByText('Train')).toBeInTheDocument();
    act(() => { useViewStore.setState({ readOnlyOverride: 'off' }); });
    expect(screen.getByRole('button', { name: 'Next step from Train' })).toBeEnabled();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(useGraphStore.getState().nodes).toEqual(originalNodes);
    expect(useGraphStore.getState().edges).toHaveLength(0);
  });
});
