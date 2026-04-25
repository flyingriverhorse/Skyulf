import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { fireEvent } from '@testing-library/react';
import { useEscapeKey } from './useEscapeKey';

describe('useEscapeKey', () => {
  it('fires the callback on Escape', () => {
    const cb = vi.fn();
    renderHook(() => useEscapeKey(cb));
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('ignores non-Escape keys', () => {
    const cb = vi.fn();
    renderHook(() => useEscapeKey(cb));
    fireEvent.keyDown(document, { key: 'Enter' });
    fireEvent.keyDown(document, { key: 'a' });
    fireEvent.keyDown(document, { key: ' ' });
    expect(cb).not.toHaveBeenCalled();
  });

  it('does nothing when enabled=false', () => {
    const cb = vi.fn();
    renderHook(() => useEscapeKey(cb, false));
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(cb).not.toHaveBeenCalled();
  });

  it('always invokes the latest callback (no stale closure)', () => {
    const first = vi.fn();
    const second = vi.fn();
    const { rerender } = renderHook(({ cb }) => useEscapeKey(cb), {
      initialProps: { cb: first },
    });
    rerender({ cb: second });
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });

  it('removes the listener on unmount', () => {
    const cb = vi.fn();
    const { unmount } = renderHook(() => useEscapeKey(cb));
    unmount();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(cb).not.toHaveBeenCalled();
  });

  it('toggles the listener when enabled flips', () => {
    const cb = vi.fn();
    const { rerender } = renderHook(({ enabled }) => useEscapeKey(cb, enabled), {
      initialProps: { enabled: true },
    });
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(cb).toHaveBeenCalledTimes(1);

    act(() => rerender({ enabled: false }));
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(cb).toHaveBeenCalledTimes(1);

    act(() => rerender({ enabled: true }));
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(cb).toHaveBeenCalledTimes(2);
  });
});
