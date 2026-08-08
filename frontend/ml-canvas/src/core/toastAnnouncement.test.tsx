import { afterEach, describe, expect, it } from 'vitest';
import { act, cleanup, render, screen } from '@testing-library/react';
import { Toaster } from 'sonner';

import { toast } from './toast';

/**
 * Mirrors the toast surface mounted in `main.tsx` so the assertions see the
 * announcements a screen reader would actually receive.
 */
const renderToastSurface = (): void => {
  render(<Toaster />);
};

const liveRegionMatches = (message: string): Element[] =>
  Array.from(document.querySelectorAll('[aria-live], [role="status"], [role="alert"]')).filter(
    (element) => element.textContent?.includes(message),
  );

describe('toast live-region announcements', () => {
  afterEach(() => {
    act(() => {
      toast.dismissAll();
    });
    cleanup();
  });

  it('announces a success toast exactly once', async () => {
    renderToastSurface();

    act(() => {
      toast.success('Dataset saved');
    });

    expect(await screen.findByText('Dataset saved')).toBeInTheDocument();
    expect(liveRegionMatches('Dataset saved')).toHaveLength(1);
  });

  it('announces an error toast exactly once', async () => {
    renderToastSurface();

    act(() => {
      toast.error('Save failed');
    });

    expect(await screen.findByText('Save failed')).toBeInTheDocument();
    expect(liveRegionMatches('Save failed')).toHaveLength(1);
  });
});
