import type { PropsWithChildren } from 'react';
import { createPortal } from 'react-dom';
import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import { Position, type EdgeProps } from '@xyflow/react';
import type { CanvasLeakageIssue } from '../../core/types/leakage';
import { CanvasLeakageContext } from '../../core/contexts/CanvasLeakageContext';
import { useViewStore } from '../../core/store/useViewStore';
import { CustomEdge } from './CustomEdge';

const controls = vi.hoisted(() => ({ deleteElements: vi.fn() }));

vi.mock('@xyflow/react', async importOriginal => ({
  ...await importOriginal<typeof import('@xyflow/react')>(),
  // The edge path and controls remain real; React Flow owns portal setup and deletion.
  EdgeLabelRenderer: ({ children }: PropsWithChildren) => createPortal(children, document.body),
  useReactFlow: () => ({ deleteElements: controls.deleteElements, getInternalNode: () => undefined }),
}));

const issue: CanvasLeakageIssue = {
  id: 'held-out-fit', nodeId: 'scaler', edgeIds: ['edge-1'], severity: 'error',
  message: 'Scaling learns from held-out rows.',
  suggestion: 'Fit scaling on the training output, then transform the test output.',
};

const edgeProps: EdgeProps = {
  id: 'edge-1', source: 'split', target: 'scaler', sourceX: 0, sourceY: 0,
  targetX: 200, targetY: 0, sourcePosition: Position.Right, targetPosition: Position.Left,
  selectable: true, deletable: true, data: { sourceLabel: 'Training data', targetLabel: 'Scaling' },
};

/** Context keeps live warnings and UI callbacks out of copyable edge data. */
function LeakageEdge({ issues, openGuide, ...props }: EdgeProps & {
  issues: CanvasLeakageIssue[]; openGuide?: () => void;
}) {
  return <CanvasLeakageContext.Provider value={{ nodeIssues: {}, edgeIssues: { 'edge-1': issues }, openGuide: openGuide ?? (() => {}) }}>
    <CustomEdge {...props} />
  </CanvasLeakageContext.Provider>;
}

beforeEach(() => {
  controls.deleteElements.mockClear();
  useViewStore.setState({ readOnlyOverride: 'off' });
});

it.each([
  { severity: 'error' as const, icon: 'text-red-600' },
  { severity: 'warning' as const, icon: 'text-amber-600' },
])('shows a persistent accessible $severity marker before hover or selection', ({ severity, icon }) => {
  /** Warnings must be discoverable without relying on hover or color alone. */
  render(<svg><LeakageEdge {...edgeProps} issues={[{ ...issue, severity }]} /></svg>);

  expect(screen.getByRole('button', { name: `Data leakage ${severity}: Training data to Scaling` })).toHaveClass(icon);
  expect(screen.getByRole('button', { name: /Remove connection/ })).toHaveStyle({ opacity: 0 });
});

it('opens advice by keyboard focus and dismisses without canvas clicks', async () => {
  /** Inspecting the warning must not select or delete canvas elements. */
  const user = userEvent.setup();
  const onCanvasClick = vi.fn();
  render(<svg onClick={onCanvasClick}><LeakageEdge {...edgeProps} issues={[issue]} /></svg>);
  await act(async () => { await user.tab(); await user.tab(); });
  const marker = screen.getByRole('button', { name: /Data leakage error/ });
  expect(marker).toHaveFocus();
  expect(screen.getByRole('dialog', { name: 'Data leakage details' })).toHaveTextContent(issue.message);
  expect(screen.getByRole('dialog')).toHaveTextContent(issue.suggestion);
  await act(async () => { await user.keyboard('{Escape}'); });
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  await act(async () => { await user.click(marker); });
  await act(async () => { await user.click(screen.getByRole('button', { name: 'Close data leakage details' })); });

  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(marker).toHaveFocus();
  expect(controls.deleteElements).not.toHaveBeenCalled();
  expect(onCanvasClick).not.toHaveBeenCalled();
});

it('opens on touch and dismisses after an outside pointer action', async () => {
  /** Touch users need persistent advice that can be dismissed without hovering. */
  const user = userEvent.setup();
  render(<svg><LeakageEdge {...edgeProps} issues={[issue]} /></svg>);
  await act(async () => {
    await user.pointer([{ keys: '[TouchA>]', target: screen.getByRole('button', { name: /Data leakage error/ }) }, { keys: '[/TouchA]' }]);
  });
  expect(screen.getByRole('dialog')).toHaveTextContent(issue.message);
  await act(async () => { await user.click(document.body); });

  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

it.each([{ readOnly: true, deletable: true }, { readOnly: false, deletable: false }])(
  'preserves warnings when readOnly=$readOnly and deletable=$deletable', ({ readOnly, deletable }) => {
    /** A disabled mutation path must not hide leakage information. */
    render(<svg><LeakageEdge {...edgeProps} issues={[issue]} deletable={deletable} data={{
      ...edgeProps.data, readOnly,
    }} /></svg>);
    fireEvent.click(screen.getByRole('button', { name: /Data leakage error/ }));

    expect(screen.queryByRole('button', { name: /Remove connection/ })).not.toBeInTheDocument();
    expect(screen.getByRole('dialog')).toHaveTextContent(issue.suggestion);
  },
);

it('keeps removal keyboard-accessible and honors a newly enabled read-only mode', async () => {
  /** Leakage controls must not break deletion or bypass the live read-only guard. */
  const user = userEvent.setup();
  render(<svg><LeakageEdge {...edgeProps} issues={[issue]} /></svg>);
  const remove = screen.getByRole('button', { name: /Remove connection/ });
  act(() => { remove.focus(); });
  expect(remove).toHaveStyle({ opacity: 1 });
  await act(async () => { await user.keyboard('{Enter}'); });
  expect(controls.deleteElements).toHaveBeenCalledWith({ edges: [{ id: 'edge-1' }] });
  controls.deleteElements.mockClear();
  useViewStore.setState({ readOnlyOverride: 'on' });
  await act(async () => { await user.keyboard('{Enter}'); });

  expect(controls.deleteElements).not.toHaveBeenCalled();
});

it('retains branch annotations and removal alongside the leakage warning', () => {
  /** Adding leakage advice must not remove merge context or the delete action. */
  render(<svg><LeakageEdge {...edgeProps} issues={[issue]} data={{
    ...edgeProps.data, branchLabel: 'Path A', branchColor: '#3b82f6', isMergeWinner: true,
  }} /></svg>);
  const marker = screen.getByRole('button', { name: /Data leakage error/ });
  expect(marker).toBeInTheDocument();
  expect(screen.getByText('Path A')).toBeInTheDocument();
  expect(screen.getByText('Wins merge')).toBeInTheDocument();
  fireEvent.focus(screen.getByRole('button', { name: /Remove connection/ }));
  expect(screen.getByRole('button', { name: /Remove connection/ })).toHaveStyle({ opacity: 1 });
});

it('clears open advice when the issue is removed and never reopens stale content', () => {
  /** Graph edits must remove both the marker and its detached popover content. */
  const { rerender } = render(<svg><LeakageEdge {...edgeProps} issues={[issue]} /></svg>);
  fireEvent.click(screen.getByRole('button', { name: /Data leakage error/ }));
  expect(screen.getByRole('dialog')).toHaveTextContent(issue.message);
  rerender(<svg><LeakageEdge {...edgeProps} issues={[]} /></svg>);

  expect(screen.queryByRole('button', { name: /Data leakage/ })).not.toBeInTheDocument();
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Remove connection/ })).toBeInTheDocument();
});

it('measures controls along the curve and searches both directions for warning spacing', () => {
  /** Tight bends need geometric separation, and a changed invalid path must use its new label. */
  const lengthDescriptor = Object.getOwnPropertyDescriptor(SVGElement.prototype, 'getTotalLength');
  const pointDescriptor = Object.getOwnPropertyDescriptor(SVGElement.prototype, 'getPointAtLength');
  const getTotalLength = vi.fn(() => 200);
  Object.defineProperty(SVGElement.prototype, 'getTotalLength', { configurable: true, value: getTotalLength });
  Object.defineProperty(SVGElement.prototype, 'getPointAtLength', { configurable: true, value: (length: number) => (
    length <= 100 ? { x: length, y: 0 } : { x: 100, y: (length - 100) / 2 }
  ) });
  try {
    const { rerender } = render(<svg><LeakageEdge {...edgeProps} issues={[issue]} /></svg>);
    expect(screen.getByRole('button', { name: /Remove connection/ }).parentElement).toHaveStyle({
      transform: 'translate(-50%, -50%) translate(100px,0px)',
    });
    expect(screen.getByRole('button', { name: /Data leakage error/ }).parentElement).toHaveStyle({
      transform: 'translate(-50%, -50%) translate(72px,0px)',
    });
    getTotalLength.mockReturnValue(0);
    rerender(<svg><LeakageEdge {...edgeProps} targetX={300} issues={[issue]} /></svg>);
    expect(screen.getByRole('button', { name: /Remove connection/ }).parentElement).toHaveStyle({
      transform: 'translate(-50%, -50%) translate(150px,0px)',
    });
    expect(screen.getByRole('button', { name: /Data leakage error/ }).parentElement).toHaveStyle({
      transform: 'translate(-50%, -50%) translate(150px,0px)',
    });
  } finally {
    if (lengthDescriptor) Object.defineProperty(SVGElement.prototype, 'getTotalLength', lengthDescriptor);
    else Reflect.deleteProperty(SVGElement.prototype, 'getTotalLength');
    if (pointDescriptor) Object.defineProperty(SVGElement.prototype, 'getPointAtLength', pointDescriptor);
    else Reflect.deleteProperty(SVGElement.prototype, 'getPointAtLength');
  }
});
