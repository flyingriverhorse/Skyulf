import { act, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { TrainingActionFooter } from './TrainingActionFooter';

afterEach(() => vi.unstubAllGlobals());

it('keeps focused details accessible until focus leaves the collapsing footer', () => {
  // Scrolling must not hide a focused history button or leave hidden controls in the tab order.
  let intersect!: (entries: { isIntersecting: boolean }[]) => void;
  const disconnect = vi.fn();
  vi.stubGlobal('IntersectionObserver', class {
    constructor(callback: typeof intersect) { intersect = callback; }
    observe() {}
    disconnect = disconnect;
  });
  const { unmount } = render(<TrainingActionFooter details={<button>History</button>}><button>Train</button></TrainingActionFooter>);
  const footer = screen.getByTestId('training-action-footer');
  const details = screen.getByText('History').closest('[aria-hidden]') as HTMLElement;
  expect(details.inert).toBe(true);
  act(() => intersect([{ isIntersecting: true }]));
  act(() => screen.getByRole('button', { name: 'History' }).focus());
  act(() => intersect([{ isIntersecting: false }]));
  expect(footer).toHaveAttribute('data-expanded', 'true');
  expect(screen.getByRole('button', { name: 'History' })).toHaveFocus();
  act(() => screen.getByRole('button', { name: 'Train' }).focus());
  expect(footer).toHaveAttribute('data-expanded', 'false');
  expect(details.inert).toBe(true);
  unmount();
  expect(disconnect).toHaveBeenCalledOnce();
});

it('keeps all controls available without intersection observation', () => {
  // Progressive enhancement must never make model history unreachable.
  vi.stubGlobal('IntersectionObserver', undefined);
  render(<TrainingActionFooter details={<button>History</button>}><button>Train</button></TrainingActionFooter>);
  expect(screen.getByTestId('training-action-footer')).toHaveAttribute('data-expanded', 'true');
  expect(screen.getByRole('button', { name: 'History' })).toBeVisible();
});
