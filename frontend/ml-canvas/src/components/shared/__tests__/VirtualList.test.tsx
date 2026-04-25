import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import { VirtualList } from '../VirtualList';

/**
 * jsdom returns 0 for layout sizes by default, which would make the
 * virtualizer think nothing is visible. Stub `getBoundingClientRect` and
 * `offset*` so a few rows render in the viewport.
 */
const stubLayout = () => {
  Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
    configurable: true,
    get: () => 600,
  });
  Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
    configurable: true,
    get: () => 400,
  });
  HTMLElement.prototype.getBoundingClientRect = function () {
    return { width: 400, height: 600, top: 0, left: 0, bottom: 600, right: 400, x: 0, y: 0, toJSON: () => ({}) };
  };
};

describe('VirtualList', () => {
  beforeEach(() => stubLayout());
  afterEach(() => cleanup());

  it('renders every row when below the threshold (cheap path)', () => {
    const items = Array.from({ length: 10 }, (_, i) => ({ id: i, label: `row-${i}` }));
    render(
      <VirtualList
        items={items}
        getKey={(it) => it.id}
        estimateSize={40}
        threshold={50}
        renderItem={(it) => <div>{it.label}</div>}
      />,
    );
    // All 10 rows should be in the DOM since we did not virtualize.
    expect(screen.getAllByText(/^row-/)).toHaveLength(10);
  });

  it('falls through to the virtualizer above the threshold', () => {
    const items = Array.from({ length: 200 }, (_, i) => ({ id: i, label: `row-${i}` }));
    render(
      <VirtualList
        items={items}
        getKey={(it) => it.id}
        estimateSize={40}
        threshold={50}
        renderItem={(it) => <div>{it.label}</div>}
      />,
    );
    // Only a slice should render; the rest must be skipped to keep the DOM small.
    const rendered = screen.queryAllByText(/^row-/);
    expect(rendered.length).toBeGreaterThan(0);
    expect(rendered.length).toBeLessThan(items.length);
  });

  it('uses the supplied key extractor to mount each row exactly once', () => {
    const items = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
    render(
      <VirtualList
        items={items}
        getKey={(it) => it.id}
        estimateSize={20}
        renderItem={(it) => <div>{it.id}</div>}
      />,
    );
    // Each id should be in the DOM once, no duplicate keys.
    expect(screen.getAllByText('a')).toHaveLength(1);
    expect(screen.getAllByText('b')).toHaveLength(1);
    expect(screen.getAllByText('c')).toHaveLength(1);
  });
});
