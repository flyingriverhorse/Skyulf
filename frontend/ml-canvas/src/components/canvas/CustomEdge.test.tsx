import type { PropsWithChildren } from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { getBezierPath, getSmoothStepPath, getStraightPath, Position, type EdgeProps } from '@xyflow/react';
import { CustomEdge } from './CustomEdge';

const controls = vi.hoisted(() => ({ deleteElements: vi.fn(), getInternalNode: vi.fn() }));

vi.mock('@xyflow/react', async importOriginal => ({
  ...await importOriginal<typeof import('@xyflow/react')>(),
  EdgeLabelRenderer: ({ children }: PropsWithChildren) => <foreignObject>{children}</foreignObject>,
  useReactFlow: () => controls,
}));

const props: EdgeProps = {
  id: 'edge', source: 'split', target: 'target', sourceX: 0, sourceY: 0,
  targetX: 200, targetY: 0, sourcePosition: Position.Right, targetPosition: Position.Left,
  selectable: true, deletable: true,
};

beforeEach(() => { controls.getInternalNode.mockReset(); });
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

it.each([
  { sourcePosition: Position.Right, targetPosition: Position.Left, targetX: 200, targetY: 5, straight: true },
  { sourcePosition: Position.Right, targetPosition: Position.Left, targetX: 200, targetY: 6, straight: false },
  { sourcePosition: Position.Bottom, targetPosition: Position.Top, targetX: 5, targetY: 200, straight: true },
  { sourcePosition: Position.Bottom, targetPosition: Position.Top, targetX: 6, targetY: 200, straight: false },
  { sourcePosition: Position.Right, targetPosition: Position.Top, targetX: 200, targetY: 0, straight: false },
])('preserves path selection for $sourcePosition to $targetPosition at ($targetX, $targetY)', ({ straight, ...geometry }) => {
  /** Nearly collinear edges must stay visible without straightening mixed axes. */
  const edge = { ...props, ...geometry };
  const { container } = render(<svg><CustomEdge {...edge} /></svg>);
  const expected = straight ? getStraightPath(edge) : getSmoothStepPath({ ...edge, borderRadius: 16 });
  expect(container.querySelector('path.react-flow__edge-path')).toHaveAttribute('d', expected[0]);
});

it('groups resolved split handles into a bounded trunk with branch paths', () => {
  /** Missing handles must not displace the junction or remove resolved branches. */
  controls.getInternalNode.mockReturnValue({ internals: {
    positionAbsolute: { x: 10, y: 20 }, handleBounds: { source: [
      { id: 'train', x: 0, y: 0, width: 10, height: 10 },
      { id: 'test', x: 0, y: 100, width: 10, height: 10 },
    ] },
  } });
  const { container } = render(<svg><CustomEdge {...props} markerEnd="url(#arrow)" data={{ splitHandles: ['train', 'missing', 'test'] }} /></svg>);
  const junction = container.querySelector('[data-split-junction]');
  expect(junction).toHaveAttribute('cx', '64');
  expect(junction).toHaveAttribute('cy', '75');
  expect(container.querySelectorAll('[data-split-handle]')).toHaveLength(2);
  expect(container.querySelector('[data-split-handle="train"] path')).toHaveAttribute('d', getBezierPath({
    sourceX: 15, sourceY: 25, sourcePosition: Position.Right, targetX: 64, targetY: 75, targetPosition: Position.Left,
  })[0]);
  expect(container.querySelector('path[marker-end]')).toHaveAttribute('d', getSmoothStepPath({
    ...props, sourceX: 64, sourceY: 75, borderRadius: 16,
  })[0]);
});

it('preserves branch style precedence and merge winner annotations', () => {
  /** Branch colors and shared dashes must survive caller styles and winner emphasis. */
  const { container, rerender } = render(<svg><CustomEdge {...props} style={{ stroke: 'red', filter: 'blur(2px)', opacity: 0.2 }} data={{
    branchColor: '#123456', branchLabel: 'Shared', branchShared: true, isMergeWinner: true,
  }} /></svg>);
  expect(container.querySelector('path.react-flow__edge-path')).toHaveStyle({ stroke: '#123456', strokeWidth: 4, strokeDasharray: '6 4', opacity: 0.7 });
  expect(container.querySelector('path.react-flow__edge-path')?.getAttribute('style')).not.toContain('blur');
  expect(screen.getByText('Wins merge')).toHaveStyle({ transform: 'translate(-50%, -100%) translate(100px,-40px)' });
  rerender(<svg><CustomEdge {...props} style={{ stroke: 'red', filter: 'blur(2px)' }} data={{ isMergeWinner: true }} /></svg>);
  expect(container.querySelector('path.react-flow__edge-path')).toHaveStyle({ stroke: '#f59e0b', strokeWidth: 4 });
});

it('keeps hover across controls and clears the pending leave timer on unmount', () => {
  /** Delayed dismissal bridges the SVG and controls without a leftover timer. */
  vi.useFakeTimers();
  const schedule = vi.spyOn(globalThis, 'setTimeout');
  const cancel = vi.spyOn(globalThis, 'clearTimeout');
  const { container, unmount } = render(<svg><CustomEdge {...props} /></svg>);
  const edge = container.querySelector('g')!;
  const remove = screen.getByRole('button', { name: 'Remove connection from Source node to Target node' });
  fireEvent.mouseEnter(edge);
  expect(remove).toHaveStyle({ opacity: 1 });
  fireEvent.mouseLeave(edge);
  act(() => { vi.advanceTimersByTime(100); });
  fireEvent.mouseEnter(remove.parentElement!);
  act(() => { vi.advanceTimersByTime(150); });
  expect(remove).toHaveStyle({ opacity: 1 });
  fireEvent.mouseLeave(remove.parentElement!);
  act(() => { vi.advanceTimersByTime(150); });
  expect(remove).toHaveStyle({ opacity: 0 });
  fireEvent.mouseLeave(edge);
  const pendingLeave = schedule.mock.results.at(-1)?.value;
  unmount();
  expect(cancel).toHaveBeenCalledWith(pendingLeave);
});
