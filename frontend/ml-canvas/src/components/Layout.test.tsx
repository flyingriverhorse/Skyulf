import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { renderToString } from 'react-dom/server';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { monitoringApi } from '../core/api/monitoring';
import { useNotificationsStore } from '../core/store/useNotificationsStore';
import { Layout } from './Layout';

vi.mock('../core/api/monitoring', () => ({
  monitoringApi: { getDriftStatus: vi.fn(), getUnresolvedCount: vi.fn() },
}));

const originalWidth = window.innerWidth;

beforeEach(() => {
  vi.useFakeTimers();
  window.innerWidth = 1280;
  vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
  document.documentElement.classList.remove('dark', 'theme-switching');
  localStorage.clear();
  useNotificationsStore.setState({ items: [] });
  vi.mocked(monitoringApi.getDriftStatus).mockReset().mockResolvedValue({
    has_drift: false, drifted_jobs: 0, unacknowledged_critical: 0,
  });
  vi.mocked(monitoringApi.getUnresolvedCount).mockReset().mockResolvedValue(0);
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  window.innerWidth = originalWidth;
  document.documentElement.classList.remove('dark', 'theme-switching');
});

/** Render the real shell and notification center with a nested route outlet. */
async function renderLayout(path = '/') {
  let view!: ReturnType<typeof render>;
  await act(async () => {
    view = render(
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="*" element={<p>Current page content</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
  });
  return view;
}

describe('Layout notification placement', () => {
  it.each([
    { width: 1280, path: '/', bell: true },
    { width: 390, path: '/', bell: true },
    { width: 1280, path: '/canvas', bell: false },
    { width: 390, path: '/canvas', bell: false },
  ])('places notifications correctly at $width px on $path', async ({ width, path, bell }) => {
    // Canvas supplies its own notification button; the shell must not duplicate it.
    window.innerWidth = width;
    await renderLayout(path);

    expect(screen.queryAllByRole('button', { name: 'Notifications' })).toHaveLength(bell ? 1 : 0);
    expect(screen.getByText('Current page content')).toBeInTheDocument();
    if (bell) {
      fireEvent.click(screen.getByRole('button', { name: 'Notifications' }));
      expect(screen.getByText(/No notifications\. Pipeline advisories/)).toBeInTheDocument();
    }
  });

  it('moves notification access when navigating out of the canvas', async () => {
    // A route change must restore access without a full page reload.
    await renderLayout('/canvas');
    expect(screen.queryByRole('button', { name: 'Notifications' })).not.toBeInTheDocument();

    await act(async () => fireEvent.click(screen.getByRole('link', { name: 'Dashboard' })));

    expect(screen.getAllByRole('button', { name: 'Notifications' })).toHaveLength(1);
    expect(screen.getByRole('link', { name: 'Dashboard' })).toHaveAttribute('aria-current', 'page');
  });
});

describe('Layout mobile navigation', () => {
  it('manages drawer focus and closes with Escape, its close button, and its backdrop', async () => {
    // Keyboard and pointer users must be able to leave the off-canvas navigation.
    window.innerWidth = 390;
    const { container } = await renderLayout();
    const opener = screen.getByRole('button', { name: 'Open navigation menu' });
    fireEvent.click(opener);
    await act(async () => vi.advanceTimersByTime(32));
    expect(screen.getByRole('link', { name: 'Dashboard' })).toHaveFocus();
    fireEvent.keyDown(window, { key: 'ArrowDown' });
    expect(opener).toHaveAttribute('aria-expanded', 'true');

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(opener).toHaveAttribute('aria-expanded', 'false');
    expect(opener).toHaveFocus();

    fireEvent.click(opener);
    fireEvent.click(screen.getByRole('button', { name: 'Close navigation menu' }));
    expect(opener).toHaveAttribute('aria-expanded', 'false');

    fireEvent.click(opener);
    const backdrop = container.querySelector('div[aria-hidden="true"]');
    expect(backdrop).not.toBeNull();
    fireEvent.click(backdrop!);
    expect(opener).toHaveAttribute('aria-expanded', 'false');
  });

  it('closes on navigation and resets when the viewport changes to desktop', async () => {
    // An old open drawer must not reappear after changing routes or screen size.
    window.innerWidth = 390;
    await renderLayout();
    fireEvent.click(screen.getByRole('button', { name: 'Open navigation menu' }));
    await act(async () => fireEvent.click(screen.getByRole('link', { name: 'EDA' })));
    expect(screen.getByRole('button', { name: 'Open navigation menu' })).toHaveAttribute('aria-expanded', 'false');

    fireEvent.click(screen.getByRole('button', { name: 'Open navigation menu' }));
    window.innerWidth = 1280;
    fireEvent.resize(window);
    await act(async () => vi.advanceTimersByTime(32));
    expect(screen.queryByRole('button', { name: 'Open navigation menu' })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'EDA' })).toHaveAttribute('title', 'EDA');

    window.innerWidth = 390;
    fireEvent.resize(window);
    await act(async () => vi.advanceTimersByTime(32));
    expect(screen.getByRole('button', { name: 'Open navigation menu' })).toHaveAttribute('aria-expanded', 'false');
  });
});

describe('Layout monitoring and theme', () => {
  it.each([
    { critical: 1, drift: false, errors: 2, badge: true },
    { critical: 0, drift: true, errors: 0, badge: true },
    { critical: 0, drift: false, errors: 0, badge: false },
  ])('shows the current monitoring state: $critical critical, drift $drift', async ({ critical, drift, errors, badge }) => {
    // Triage-worthy alerts and unresolved errors must remain visible in navigation.
    vi.mocked(monitoringApi.getDriftStatus).mockResolvedValue({
      has_drift: drift, drifted_jobs: drift ? 1 : 0, unacknowledged_critical: critical,
    });
    vi.mocked(monitoringApi.getUnresolvedCount).mockResolvedValue(errors);
    await renderLayout();

    expect(screen.getByRole('link', { name: 'Data Drift' }).querySelector('.bg-red-500') !== null).toBe(badge);
    expect(screen.getByRole('link', { name: 'Error Log' }).querySelector('.bg-red-500') !== null).toBe(errors > 0);
  });

  it('preserves navigation when monitoring requests fail', async () => {
    // Badge enrichment must not make the application shell unusable offline.
    vi.mocked(monitoringApi.getDriftStatus).mockRejectedValue(new Error('offline'));
    vi.mocked(monitoringApi.getUnresolvedCount).mockRejectedValue(new Error('offline'));
    await renderLayout();

    expect(screen.getByText('Current page content')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Data Drift' }).querySelector('.bg-red-500')).toBeNull();
  });

  it('polls only visible non-error pages and stops polling after unmount', async () => {
    // Background tabs, the live error page, and unmounted shells must not keep polling.
    const { unmount } = await renderLayout();
    expect(monitoringApi.getUnresolvedCount).toHaveBeenCalledTimes(1);
    vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
    await act(async () => vi.advanceTimersByTime(300_000));
    expect(monitoringApi.getUnresolvedCount).toHaveBeenCalledTimes(1);
    vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
    await act(async () => fireEvent.click(screen.getByRole('link', { name: 'Error Log' })));
    await act(async () => vi.advanceTimersByTime(300_000));
    expect(monitoringApi.getUnresolvedCount).toHaveBeenCalledTimes(1);
    await act(async () => fireEvent.click(screen.getByRole('link', { name: 'Dashboard' })));
    expect(monitoringApi.getUnresolvedCount).toHaveBeenCalledTimes(2);
    unmount();
    await act(async () => vi.advanceTimersByTime(300_000));
    expect(monitoringApi.getUnresolvedCount).toHaveBeenCalledTimes(2);
  });

  it('starts from the existing theme and persists both toggle directions', async () => {
    // The shell must respect the pre-paint theme and retain subsequent choices.
    document.documentElement.classList.add('dark');
    await renderLayout();
    fireEvent.click(screen.getByRole('button', { name: 'Switch to light mode' }));
    expect(document.documentElement).not.toHaveClass('dark');
    expect(localStorage.getItem('skyulf-theme')).toBe('light');
    fireEvent.click(screen.getByRole('button', { name: 'Switch to dark mode' }));
    expect(localStorage.getItem('skyulf-theme')).toBe('dark');
    expect(document.documentElement).toHaveClass('dark');
  });

  it('can render without a browser document', () => {
    // Server rendering must use the light-theme fallback without touching document.
    let html: string;
    vi.stubGlobal('document', undefined);
    try {
      html = renderToString(<MemoryRouter initialEntries={['/canvas']}><Layout /></MemoryRouter>);
    } finally {
      vi.unstubAllGlobals();
    }

    expect(html).toContain('Switch to dark mode');
  });
});
