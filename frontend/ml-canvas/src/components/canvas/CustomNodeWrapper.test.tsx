import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { Edge, NodeProps } from '@xyflow/react';
import type { NodeDefinition } from '../../core/types/nodes';
import type { CanvasLeakageIssue } from '../../core/types/leakage';
import { CanvasLeakageContext } from '../../core/contexts/CanvasLeakageContext';
import { useViewStore } from '../../core/store/useViewStore';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useJobStore } from '../../core/store/useJobStore';
import { CustomNodeWrapper } from './CustomNodeWrapper';

const controls = vi.hoisted(() => ({ getDefinition: vi.fn(), deleteElements: vi.fn(), getEdges: vi.fn<() => Edge[]>(() => []) }));

vi.mock('@xyflow/react', () => ({
  Handle: ({ children, isConnectable, 'aria-label': label, title }: React.PropsWithChildren<{
    isConnectable?: boolean; 'aria-label'?: string; title?: string;
  }>) => <div aria-label={label} title={title} data-connectable={isConnectable}>{children}</div>,
  Position: { Left: 'left', Right: 'right' },
  useConnection: (selector: (state: { fromHandle: null }) => unknown) => selector({ fromHandle: null }),
  useReactFlow: () => ({ deleteElements: controls.deleteElements, getEdges: controls.getEdges }),
}));
vi.mock('./ConnectionPicker', () => ({
  ConnectionPicker: ({ port }: { port: { label: string } }) => <button>{port.label}</button>,
}));
vi.mock('../../core/registry/NodeRegistry', () => ({
  registry: { get: controls.getDefinition },
}));

/** A complete registry entry keeps the wrapper under test independent of global registration. */
function definition(overrides: Partial<NodeDefinition<unknown>> = {}): NodeDefinition<unknown> {
  return {
    type: 'feature_target_split', label: 'X/Y Split', category: 'Preprocessing',
    description: 'Split feature and target columns.', inputs: [],
    outputs: [{ id: 'X', label: 'Features (X)', type: 'dataset' }, { id: 'y', label: 'Target (y)', type: 'dataset' }],
    validate: () => ({ isValid: true }), bodyPreview: () => 'Select target',
    settings: () => null, getDefaultConfig: () => ({}), ...overrides,
  };
}

const nodeProps: NodeProps = {
  id: 'split', data: { definitionType: 'feature_target_split' }, selected: false, isConnectable: true,
  type: 'custom', draggable: true, zIndex: 0, dragging: false, selectable: true, deletable: true,
  positionAbsoluteX: 0, positionAbsoluteY: 0,
};

beforeEach(() => {
  controls.getDefinition.mockReset().mockReturnValue(definition());
  controls.deleteElements.mockClear();
  controls.getEdges.mockReset().mockReturnValue([]);
  useViewStore.setState({ readOnlyOverride: 'off', perfOverlayEnabled: false });
  useJobStore.setState({ jobs: [] });
  useGraphStore.setState({ executionResult: null, nodeJobSummaries: {}, branchEdgeLabels: {},
    predictedSchemas: {}, brokenSchemaRefs: {}, incomingSourceCounts: {} });
});

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it('prefers inline execution summaries and telemetry over completed job summaries', () => {
  // The canvas must show the freshest preview, while retaining precise zero-valued metrics.
  controls.getDefinition.mockReturnValue(definition({ outputs: [] }));
  useGraphStore.setState({
    executionResult: { pipeline_id: 'preview', status: 'success', preview_data: null, recommendations: [],
      node_results: { split: { status: 'success', execution_time: 0.25,
        metadata: { summary: '  Fresh preview  ' },
        metrics: { fit_time: 0, peak_memory_bytes: 0, rows_in: 10, rows_out: 0 } } } },
    nodeJobSummaries: { split: [{ summary: 'Older job', branch_index: 0, pipeline_id: 'job',
      parent_pipeline_id: null, finished_at: null, duration_ms: 9999 }] },
  });
  useViewStore.setState({ perfOverlayEnabled: true });
  render(<CustomNodeWrapper {...nodeProps} />);
  expect(screen.getByText('Fresh preview')).toBeInTheDocument();
  expect(screen.queryByText('Older job')).not.toBeInTheDocument();
  expect(screen.getByTestId('canvas-node-feature_target_split')).toHaveAttribute('data-perf-duration-ms', '250');
  expect(screen.getByTitle('Core fit time')).toHaveTextContent('0ms');
  expect(screen.getByTitle('Peak Memory')).toHaveTextContent('0.0MB');
  expect(screen.getByTestId('canvas-node-feature_target_split')).toHaveAttribute('title',
    'Last run: 250ms\nFit time: 0ms\nPeak mem: 0.0 MB\nRows: 10 → 0');
});

it('keeps a validation pulse tied to the invalid transition and clears it on recovery', () => {
  // Rendering an invalid node again must not restart its one-shot attention animation.
  vi.useFakeTimers();
  try {
    controls.getDefinition.mockReturnValue(definition({ outputs: [],
      validate: () => ({ isValid: false, message: 'Pick columns' }) }));
    const { rerender } = render(<CustomNodeWrapper {...nodeProps} />);
    expect(screen.getByTestId('canvas-node-feature_target_split')).toHaveClass('animate-validation-pulse');
    act(() => { vi.advanceTimersByTime(4800); });
    rerender(<CustomNodeWrapper {...nodeProps} selected />);
    expect(screen.getByTestId('canvas-node-feature_target_split')).not.toHaveClass('animate-validation-pulse');
    controls.getDefinition.mockReturnValue(definition({ outputs: [] }));
    rerender(<CustomNodeWrapper {...nodeProps} data={{ ...nodeProps.data, columns: ['x'] }} />);
    expect(screen.queryByLabelText('Configuration issue: Pick columns')).not.toBeInTheDocument();
  } finally {
    vi.useRealTimers();
  }
});

it('pairs branch summaries with sorted incoming handles and retains missing-label fallbacks', () => {
  // Card letters must match canvas edge labels even when edges arrive in a different order.
  controls.getDefinition.mockReturnValue(definition({ outputs: [] }));
  controls.getEdges.mockReturnValue([
    { id: 'b', source: 'second', target: 'split', sourceHandle: 'b' },
    { id: 'unrelated', source: 'other', target: 'elsewhere', sourceHandle: '0' },
    { id: 'a', source: 'first', target: 'split', sourceHandle: 'a' },
  ]);
  useGraphStore.setState({
    branchEdgeLabels: { a: 'Path D · Train', b: 'Path C · Test' },
    nodeJobSummaries: { split: ['First result', 'Second result', 'Third result'].map((summary, index) => ({
      summary, branch_index: index + 2, pipeline_id: `job-${index}`, parent_pipeline_id: 'parent', finished_at: null,
    })) },
  });
  useJobStore.setState({ jobs: [{ job_id: 'fresh', pipeline_id: 'fresh', node_id: 'split', job_type: 'training',
    status: 'queued', start_time: null, end_time: null, error: null, result: null, created_at: '2026-09-09' }] });
  render(<CustomNodeWrapper {...nodeProps} />);
  expect(screen.getByText('First result').parentElement).toHaveTextContent('DFirst result');
  expect(screen.getByText('Second result').parentElement).toHaveTextContent('CSecond result');
  expect(screen.getByText('Third result').parentElement).toHaveTextContent('EThird result');
  expect(screen.getByText('First result').parentElement!.parentElement).toHaveAttribute('title',
    'Previous run · new run in progress—\nPath D · Train: First result\nPath C · Test: Second result\nPath E: Third result');
});

it('reserves the widest output label in unscaled pixels and updates after font changes', () => {
  // Different platform fonts and canvas zoom must not let split labels cover the summary.
  let labelWidth = 125;
  vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockImplementation(function (this: HTMLElement) {
    return this.textContent === 'Features (X)' ? labelWidth : 80;
  });
  let resize: (() => void) | undefined;
  const observe = vi.fn();
  const disconnect = vi.fn();
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback: () => void) { resize = callback; }
    observe = observe;
    disconnect = disconnect;
  });
  const { unmount } = render(<CustomNodeWrapper {...{
    id: 'split', data: { definitionType: 'feature_target_split' }, selected: false, isConnectable: true,
    type: 'custom', draggable: true, zIndex: 0, dragging: false, selectable: true, deletable: true,
    positionAbsoluteX: 0, positionAbsoluteY: 0,
  }} />);
  const body = screen.getByText('Select target').parentElement!.parentElement!;
  expect(body.style.getPropertyValue('--split-output-space')).toBe('149px');
  expect(observe).toHaveBeenCalledTimes(2);
  act(() => { labelWidth = 170; resize?.(); });
  expect(body.style.getPropertyValue('--split-output-space')).toBe('194px');
  act(() => { labelWidth = 95; resize?.(); });
  expect(body.style.getPropertyValue('--split-output-space')).toBe('119px');
  unmount();
  expect(disconnect).toHaveBeenCalledOnce();
});

it.each([0, 1])('does not reserve split-label space for a node with %i outputs', (outputCount) => {
  // Ordinary and terminal nodes must not install a split-label resize observer.
  const constructed = vi.fn();
  vi.stubGlobal('ResizeObserver', class {
    constructor() { constructed(); }
    observe() {}
    disconnect() {}
  });
  controls.getDefinition.mockReturnValue(definition({
    outputs: outputCount ? [{ id: 'out', label: 'Output', type: 'dataset' }] : [],
  }));
  render(<CustomNodeWrapper {...nodeProps} />);

  expect(screen.getByText('Select target')).toBeInTheDocument();
  expect(screen.queryAllByRole('button', { name: 'Output' })).toHaveLength(outputCount);
  expect(constructed).not.toHaveBeenCalled();
});

it('renders an unknown node without trying to measure missing output labels', () => {
  // Older canvases can contain unregistered types and must still be inspectable.
  controls.getDefinition.mockReturnValue(undefined);
  render(<CustomNodeWrapper {...nodeProps} data={{ definitionType: 'retired_node' }} />);

  expect(screen.getByText('Unknown Node')).toBeInTheDocument();
  expect(screen.getByText('Type: retired_node')).toBeInTheDocument();
});

it.each([undefined, 0, 0.2])('only exposes validation connections when validation_size is positive: %s', (size) => {
  // A disabled validation split must not invite connections to nonexistent data.
  controls.getDefinition.mockReturnValue(definition({ outputs: [
    { id: 'train', label: 'Train', type: 'dataset' },
    { id: 'test', label: 'Test', type: 'dataset' },
    { id: 'validation', label: 'Validation', type: 'dataset' },
  ] }));
  render(<CustomNodeWrapper {...nodeProps} data={{ definitionType: 'TrainTestSplitter', validation_size: size }} />);

  expect(screen.getByLabelText('Validation output')).toHaveAttribute('data-connectable', String((size ?? 0) > 0));
  expect(screen.queryAllByRole('button', { name: 'Validation' })).toHaveLength((size ?? 0) > 0 ? 1 : 0);
});

it.each([
  { readOnly: true, connectable: true },
  { readOnly: false, connectable: false },
  { readOnly: false, connectable: true },
])('respects readOnly=$readOnly and connectable=$connectable on both port directions', ({ readOnly, connectable }) => {
  // Input and output handles must agree with the canvas editing permissions.
  useViewStore.setState({ readOnlyOverride: readOnly ? 'on' : 'off' });
  controls.getDefinition.mockReturnValue(definition({ inputs: [{ id: 'input', label: 'Input', type: 'dataset' }] }));
  render(<CustomNodeWrapper {...nodeProps} isConnectable={connectable} />);

  expect(screen.getByLabelText('Input input')).toHaveAttribute('data-connectable', String(!readOnly && connectable));
  expect(screen.getByLabelText('Features (X) output')).toHaveAttribute('data-connectable', String(!readOnly && connectable));
  expect(screen.queryAllByRole('button', { name: 'Remove node' })).toHaveLength(readOnly ? 0 : 1);
});

it('deletes the selected node through React Flow', () => {
  // The floating remove action must target this card instead of another selection.
  render(<CustomNodeWrapper {...nodeProps} selected />);
  fireEvent.click(screen.getByRole('button', { name: 'Remove node' }));

  expect(controls.deleteElements).toHaveBeenCalledWith({ nodes: [{ id: 'split' }] });
});

it.each(['missing', 'empty', 'throws'] as const)('falls back to the description when a preview is %s', (mode) => {
  // A missing or broken preview must not leave a known node unreadable.
  const bodyPreview = mode === 'missing' ? undefined : mode === 'empty' ? () => '' : () => { throw new Error('preview failed'); };
  const nodeDefinition = definition();
  if (bodyPreview) nodeDefinition.bodyPreview = bodyPreview;
  else delete nodeDefinition.bodyPreview;
  controls.getDefinition.mockReturnValue(nodeDefinition);
  render(<CustomNodeWrapper {...nodeProps} />);

  expect(screen.getByText('Split feature and target columns.')).toBeInTheDocument();
});

const leakageIssue: CanvasLeakageIssue = {
  id: 'fit-before-split', nodeId: 'split', edgeIds: [], severity: 'error',
  message: 'Scaling learns from held-out rows.',
  suggestion: 'Split first, then fit scaling on the training output.',
};

/** Runtime feedback is supplied outside graph data so copied nodes remain serializable. */
function LeakageNode({ issues, openGuide, ...props }: NodeProps & {
  issues: CanvasLeakageIssue[]; openGuide?: () => void;
}) {
  return <CanvasLeakageContext.Provider value={{ nodeIssues: { split: issues }, edgeIssues: {}, openGuide: openGuide ?? (() => {}) }}>
    <CustomNodeWrapper {...props} />
  </CanvasLeakageContext.Provider>;
}

it.each([
  { severity: 'error' as const, border: 'border-red-500', icon: 'text-red-600' },
  { severity: 'warning' as const, border: 'border-amber-500', icon: 'text-amber-600' },
])('keeps a visible $severity marker on a selected node without related edges', ({ severity, border, icon }) => {
  /** A node must identify the risk even when no connecting edge can be marked. */
  render(<LeakageNode {...nodeProps} selected issues={[{ ...leakageIssue, severity }]} />);

  expect(screen.getByTestId('canvas-node-feature_target_split')).toHaveClass(border, 'scale-[1.02]');
  expect(screen.getByRole('button', { name: `Data leakage ${severity}: X/Y Split` })).toHaveClass(icon);
});

it('keeps leakage advice available and dismissible in read-only mode', () => {
  /** Inspecting leakage must remain possible when canvas mutation is disabled. */
  useViewStore.setState({ readOnlyOverride: 'on' });
  render(<LeakageNode {...nodeProps} issues={[leakageIssue]} />);
  fireEvent.click(screen.getByRole('button', { name: 'Data leakage error: X/Y Split' }));

  expect(screen.queryByRole('button', { name: 'Remove node' })).not.toBeInTheDocument();
  expect(screen.getByRole('dialog', { name: 'Data leakage details' })).toHaveTextContent(leakageIssue.suggestion);
  fireEvent.click(screen.getByRole('button', { name: 'Close data leakage details' }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

it('removes an open leakage popover and restores selection styling when issues clear', () => {
  /** Rewiring the graph must clear old warnings immediately without leaving a stale portal. */
  const { rerender } = render(<LeakageNode {...nodeProps} selected issues={[leakageIssue]} />);
  fireEvent.focus(screen.getByRole('button', { name: 'Data leakage error: X/Y Split' }));
  expect(screen.getByRole('dialog')).toHaveTextContent(leakageIssue.message);
  rerender(<LeakageNode {...nodeProps} selected issues={[]} />);

  expect(screen.queryByRole('button', { name: /Data leakage/ })).not.toBeInTheDocument();
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.getByTestId('canvas-node-feature_target_split')).toHaveClass('border-primary');
  expect(screen.getByTestId('canvas-node-feature_target_split')).not.toHaveClass('border-red-500');
});
