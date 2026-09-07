import { fireEvent, render, screen } from '@testing-library/react';
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { ReactFlowProvider } from '@xyflow/react';
import { initializeRegistry } from '../../core/registry/init';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { ConnectionGuidance } from './ConnectionGuidance';
import { ConnectionPort } from './ConnectionPort';
import { ConnectionHoverCard } from './ConnectionHoverCard';

type DragHandle = { nodeId: string; id: string; type: 'source' | 'target' };
const drag = vi.hoisted(() => ({
  inProgress: false,
  fromHandle: null as DragHandle | null,
  toHandle: null as DragHandle | null,
  pointer: { x: 120, y: 80 },
}));
vi.mock('@xyflow/react', async importOriginal => ({
  ...await importOriginal<typeof import('@xyflow/react')>(),
  // Pointer geometry belongs to React Flow; validation, stores, ports, and popovers remain real.
  useConnection: (selector?: (state: typeof drag) => unknown) => selector ? selector(drag) : drag,
}));

beforeAll(() => { initializeRegistry(); });
beforeEach(() => {
  drag.inProgress = false;
  drag.fromHandle = null;
  drag.toHandle = null;
  useViewStore.setState({ readOnlyOverride: 'off' });
  useGraphStore.setState({ nodes: [
    { id: 'model', position: { x: 0, y: 0 }, data: { definitionType: 'classification' } },
    { id: 'clean', position: { x: 300, y: 0 }, data: { definitionType: 'imputation_node' } },
    { id: 'ensemble', position: { x: 300, y: 200 }, data: { definitionType: 'EnsembleNode' } },
  ], edges: [] });
});

describe('connection feedback', () => {
  it('shows rejection guidance in either drag direction and removes it for compatible destinations', async () => {
    // Reverse dragging must use the same model/data contract as dragging from an output.
    const { rerender } = render(<ConnectionGuidance />);
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    drag.inProgress = true;
    drag.fromHandle = { nodeId: 'model', id: 'model', type: 'source' };
    rerender(<ConnectionGuidance />);
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    drag.toHandle = { nodeId: 'clean', id: 'in', type: 'target' };
    rerender(<ConnectionGuidance />);
    expect(await screen.findByRole('tooltip', { name: 'Connection guidance' })).toHaveTextContent('trained model');
    [drag.fromHandle, drag.toHandle] = [drag.toHandle, drag.fromHandle];
    rerender(<ConnectionGuidance />);
    expect(screen.getByRole('tooltip')).toHaveTextContent('trained model');
    drag.fromHandle = { nodeId: 'ensemble', id: 'in', type: 'target' };
    rerender(<ConnectionGuidance />);
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  it('explains same-direction handles without changing keyboard focus', async () => {
    // Pointer-only advice must never steal focus from an edited field.
    drag.inProgress = true;
    drag.fromHandle = { nodeId: 'clean', id: 'out', type: 'source' };
    drag.toHandle = { nodeId: 'model', id: 'model', type: 'source' };
    const { rerender } = render(<input aria-label="Current setting" />);
    screen.getByRole('textbox').focus();
    rerender(<><input aria-label="Current setting" /><ConnectionGuidance /></>);
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Choose an input for an output');
    expect(screen.getByRole('textbox')).toHaveFocus();
  });

  it('marks compatible and incompatible inputs using real graph validation', () => {
    // The visible port ring must agree with the committed graph's endpoint rules.
    drag.fromHandle = { nodeId: 'model', id: 'model', type: 'source' };
    render(<ReactFlowProvider>
      <ConnectionPort nodeId="clean" direction="target" port={{ id: 'in', label: 'Data', type: 'dataset' }} top="50%" />
      <ConnectionPort nodeId="ensemble" direction="target" port={{ id: 'in', label: 'Data / Models', type: 'any' }} top="50%" />
    </ReactFlowProvider>);
    expect(screen.getByLabelText('Data input')).toHaveAttribute('data-connection-state', 'incompatible');
    expect(screen.getByLabelText('Data / Models input')).toHaveAttribute('data-connection-state', 'compatible');
    expect(screen.getByLabelText('Data input')).toHaveAttribute('title', expect.stringContaining('trained model'));
  });

  it('offers an editable output picker but disables validation and read-only handles', () => {
    // Read-only and inactive ports must not expose a mutation path through the plus button.
    render(<ReactFlowProvider>
      <ConnectionPort nodeId="clean" direction="source" port={{ id: 'out', label: 'Cleaned Data', type: 'dataset' }} top="50%" />
      <ConnectionPort nodeId="split" direction="source" port={{ id: 'validation', label: 'Validation', type: 'dataset' }} top="50%" disabled compact />
      <ConnectionPort nodeId="model" direction="source" port={{ id: 'model', label: 'Trained Model', type: 'model' }} top="50%" canConnect={false} />
    </ReactFlowProvider>);
    expect(screen.getByRole('button', { name: 'Next step from Cleaned Data' })).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Next step from Validation' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Validation output')).not.toHaveClass('connectable');
    expect(screen.getByLabelText('Validation output')).toHaveAttribute('title', expect.stringContaining('Validation is disabled'));
    expect(screen.getByLabelText('Trained Model output')).not.toHaveClass('connectable');
    expect(screen.queryByRole('button', { name: 'Next step from Trained Model' })).not.toBeInTheDocument();
  });

  it('keeps endpoint details readable without passing pointer actions to the canvas', async () => {
    // Inspecting a connection must not trigger a canvas click or select a node underneath it.
    const onCanvasClick = vi.fn();
    const onEnter = vi.fn();
    const onLeave = vi.fn();
    render(<button type="button" aria-label="Canvas" onClick={onCanvasClick}><ConnectionHoverCard id="wire-details" x={100} y={80}
      sourceLabel="Train-Test Split (1)" targetLabel="Scaling (1)" onEnter={onEnter} onLeave={onLeave} /></button>);
    const details = await screen.findByRole('tooltip', { name: 'Train-Test Split (1) → Scaling (1)' });
    fireEvent.mouseEnter(details);
    fireEvent.click(details);
    fireEvent.mouseLeave(details);
    expect(onEnter).toHaveBeenCalledOnce();
    expect(onLeave).toHaveBeenCalledOnce();
    expect(onCanvasClick).not.toHaveBeenCalled();
  });
});
