import { act, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { CustomNodeWrapper } from './CustomNodeWrapper';

vi.mock('@xyflow/react', () => ({
  Handle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  Position: { Left: 'left', Right: 'right' },
  useConnection: (selector: (state: { fromHandle: null }) => unknown) => selector({ fromHandle: null }),
  useReactFlow: () => ({ deleteElements: vi.fn(), getEdges: () => [] }),
}));
vi.mock('./ConnectionPicker', () => ({
  ConnectionPicker: ({ port }: { port: { label: string } }) => <button>{port.label}</button>,
}));
vi.mock('../../core/registry/NodeRegistry', () => ({
  registry: { get: () => ({
    label: 'X/Y Split', inputs: [],
    outputs: [{ id: 'X', label: 'Features (X)', type: 'dataset' }, { id: 'y', label: 'Target (y)', type: 'dataset' }],
    validate: () => ({ isValid: true }),
    bodyPreview: () => 'Select target',
  }) },
}));

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

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
