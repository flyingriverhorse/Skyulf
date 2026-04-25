import { describe, it, expect, vi } from 'vitest';
import { render, act } from '@testing-library/react';
import React from 'react';
import { useElementSize } from './useElementSize';

// Capture the most recent ResizeObserver instance + its callback so
// tests can synthesise resize entries without a layout engine.
class MockResizeObserver {
  static last: MockResizeObserver | null = null;
  cb: ResizeObserverCallback;
  observed: Element[] = [];
  constructor(cb: ResizeObserverCallback) {
    this.cb = cb;
    MockResizeObserver.last = this;
  }
  observe(el: Element) {
    this.observed.push(el);
  }
  unobserve() {}
  disconnect = vi.fn();
  trigger(width: number, height: number) {
    const entry = {
      contentRect: { width, height, top: 0, left: 0, bottom: height, right: width, x: 0, y: 0 },
      target: this.observed[0],
    } as unknown as ResizeObserverEntry;
    this.cb([entry], this as unknown as ResizeObserver);
  }
}

const installMock = () => {
  const original = window.ResizeObserver;
  (window as unknown as { ResizeObserver: typeof MockResizeObserver }).ResizeObserver =
    MockResizeObserver;
  return () => {
    window.ResizeObserver = original;
    MockResizeObserver.last = null;
  };
};

const Probe: React.FC<{ onSize: (s: { width: number; height: number }) => void }> = ({ onSize }) => {
  const [setRef, size] = useElementSize<HTMLDivElement>();
  React.useEffect(() => {
    onSize(size);
  }, [size, onSize]);
  return <div ref={setRef} data-testid="probe" />;
};

describe('useElementSize', () => {
  it('starts with width=0 and height=0', () => {
    const restore = installMock();
    const sizes: Array<{ width: number; height: number }> = [];
    render(<Probe onSize={(s) => sizes.push(s)} />);
    expect(sizes[0]).toEqual({ width: 0, height: 0 });
    restore();
  });

  it('updates the returned size when ResizeObserver fires an entry', () => {
    const restore = installMock();
    const sizes: Array<{ width: number; height: number }> = [];
    render(<Probe onSize={(s) => sizes.push(s)} />);
    act(() => {
      MockResizeObserver.last?.trigger(640, 480);
    });
    expect(sizes.at(-1)).toEqual({ width: 640, height: 480 });
    restore();
  });

  it('disconnects the observer on unmount', () => {
    const restore = installMock();
    const { unmount } = render(<Probe onSize={() => {}} />);
    const observer = MockResizeObserver.last;
    expect(observer?.disconnect).not.toHaveBeenCalled();
    unmount();
    expect(observer?.disconnect).toHaveBeenCalledOnce();
    restore();
  });
});
