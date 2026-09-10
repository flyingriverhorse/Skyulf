import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { renderToString } from 'react-dom/server';
import { useState } from 'react';
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
async function renderLayout(path = '/', content = <p>Current page content</p>) {
  let view!: ReturnType<typeof render>;
  await act(async () => {
    view = render(
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="*" element={content} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
  });
  return view;
}

describe('Layout routes and content', () => {
  it('keeps all route links in order and marks only exact routes active', async () => {
    // Navigation destinations and active state must survive presentation extraction.
    await renderLayout('/jobs');
    const routes = [
      ['Dashboard', '/'], ['Jobs', '/jobs'], ['EDA', '/eda'], ['Data Drift', '/drift'],
      ['ML Canvas', '/canvas'], ['Data Sources', '/data'], ['Model Registry', '/registry'],
      ['Deployments', '/deployments'], ['Error Log', '/errors'], ['Slow Nodes', '/slow-nodes'],
      ['Audit Log', '/audit'],
    ];
    expect(screen.getAllByRole('link').map(link => [link.textContent, link.getAttribute('href')])).toEqual(routes);
    expect(screen.getAllByRole('link').filter(link => link.hasAttribute('aria-current'))).toEqual([
      screen.getByRole('link', { name: 'Jobs' }),
    ]);
    await act(async () => fireEvent.click(screen.getByRole('link', { name: 'Data Sources' })));
    expect(screen.getByRole('link', { name: 'Jobs' })).not.toHaveAttribute('aria-current');
    expect(screen.getByRole('link', { name: 'Data Sources' })).toHaveAttribute('aria-current', 'page');
  });

  it.each(['/canvas', '/eda'])('collapses the desktop rail on %s and expands it on mobile', async path => {
    // The desktop rail retains accessible labels while mobile keeps full-width navigation.
    const { container } = await renderLayout(path);
    expect(container.querySelector('aside')).toHaveClass('w-16');
    expect(screen.queryByText('Skyulf ML')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Dashboard' })).toHaveAttribute('title', 'Dashboard');
    expect(screen.getByRole('link', { name: 'Dashboard' })).toHaveTextContent('');
    window.innerWidth = 390;
    fireEvent.resize(window);
    await act(async () => vi.advanceTimersByTime(32));
    expect(screen.getByRole('dialog', { name: 'Navigation menu' })).toHaveClass('w-64', '-translate-x-full');
    expect(screen.getByRole('dialog', { name: 'Navigation menu' })).toHaveAttribute('aria-modal', 'true');
    expect(screen.getByRole('link', { name: 'Dashboard' })).not.toHaveAttribute('title');
    expect(screen.getByRole('link', { name: 'Dashboard' })).toHaveTextContent('Dashboard');
  });

  it('keeps outlet state through navigation, theme, drawer, and viewport changes', async () => {
    // Shell updates must not remount the active route content.
    function StatefulPage() {
      const [count, setCount] = useState(0);
      return <button onClick={() => setCount(count + 1)}>Page count {count}</button>;
    }
    await renderLayout('/', <StatefulPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Page count 0' }));
    fireEvent.click(screen.getByRole('button', { name: 'Switch to dark mode' }));
    await act(async () => fireEvent.click(screen.getByRole('link', { name: 'ML Canvas' })));
    window.innerWidth = 390;
    fireEvent.resize(window);
    await act(async () => vi.advanceTimersByTime(32));
    fireEvent.click(screen.getByRole('button', { name: 'Open navigation menu' }));
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.getByRole('button', { name: 'Page count 1' })).toBeInTheDocument();
  });
});

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
  it('polls every five minutes and retains the previous badge after a failed refresh', async () => {
    // Failed enrichment must not clear a previously visible unresolved-error signal.
    vi.mocked(monitoringApi.getUnresolvedCount).mockResolvedValueOnce(2).mockRejectedValueOnce(new Error('offline')).mockResolvedValue(0);
    await renderLayout();
    expect(monitoringApi.getUnresolvedCount).toHaveBeenCalledTimes(1);
    await act(async () => vi.advanceTimersByTime(299_999));
    expect(monitoringApi.getUnresolvedCount).toHaveBeenCalledTimes(1);
    await act(async () => vi.advanceTimersByTime(1));
    expect(monitoringApi.getUnresolvedCount).toHaveBeenCalledTimes(2);
    expect(screen.getByRole('link', { name: 'Error Log' }).querySelector('.bg-red-500')).not.toBeNull();
    await act(async () => vi.advanceTimersByTime(300_000));
    expect(monitoringApi.getUnresolvedCount).toHaveBeenCalledTimes(3);
    expect(screen.getByRole('link', { name: 'Error Log' }).querySelector('.bg-red-500')).toBeNull();
    expect(monitoringApi.getDriftStatus).toHaveBeenCalledTimes(1);
  });

  it.each([{ hidden: true, path: '/' }, { hidden: false, path: '/errors' }])(
    'suppresses the initial error request when hidden=$hidden on $path', async ({ hidden, path }) => {
      // Suppression applies to the initial check as well as later interval ticks.
      vi.spyOn(document, 'hidden', 'get').mockReturnValue(hidden);
      await renderLayout(path);
      await act(async () => vi.advanceTimersByTime(300_000));
      expect(monitoringApi.getUnresolvedCount).not.toHaveBeenCalled();
      expect(monitoringApi.getDriftStatus).toHaveBeenCalledTimes(1);
    },
  );

  it('allows an in-flight error request to update the badge after navigating to errors', async () => {
    // Route cleanup cancels polling but intentionally leaves the current request alive.
    let resolveCount!: (value: number) => void;
    vi.mocked(monitoringApi.getUnresolvedCount).mockReturnValue(new Promise(resolve => { resolveCount = resolve; }));
    await renderLayout();
    await act(async () => fireEvent.click(screen.getByRole('link', { name: 'Error Log' })));
    await act(async () => resolveCount(2));
    expect(monitoringApi.getUnresolvedCount).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('link', { name: 'Error Log' }).querySelector('.bg-red-500')).not.toBeNull();
  });

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
