import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { NotificationCenter } from './NotificationCenter';
import { useNotificationsStore } from '../../core/store/useNotificationsStore';
import { useViewStore } from '../../core/store/useViewStore';
import { useJobStore } from '../../core/store/useJobStore';
import type { JobInfo } from '../../core/api/jobs';

vi.mock('../../core/toast', () => ({
  toast: {
    dismissAll: vi.fn(),
  },
}));

let originalRect: typeof Element.prototype.getBoundingClientRect;
const originalFetchJobs = useJobStore.getState().fetchJobs;

beforeEach(() => {
  originalRect = Element.prototype.getBoundingClientRect;
  Element.prototype.getBoundingClientRect = function () {
    return { width: 100, height: 20, top: 0, left: 0, bottom: 20, right: 100, x: 0, y: 0, toJSON: () => ({}) } as DOMRect;
  };
  act(() => {
    useNotificationsStore.getState().clear();
    useNotificationsStore.getState().addMany([
      {
        node_id: 'node-1',
        node_type: 'TrainingNode',
        level: 'warning',
        logger: 'pipeline',
        message: 'Training took too long',
      },
    ]);
  });
});

/** Expose the router destination so execution actions are tested through navigation. */
function LocationProbe() {
  return <span data-testid="location">{useLocation().pathname}</span>;
}

describe('NotificationCenter execution entries', () => {
  it.each(['experiments', 'inference'] as const)('switches the internal %s view to canvas for preview feedback', (activeView) => {
    // These tabs share /canvas, so router navigation alone cannot reveal preview results.
    useNotificationsStore.getState().upsertExecution('canvas-preview', 'Preview failed', { type: 'preview' });
    useViewStore.setState({ activeView, isResultsPanelExpanded: false });
    render(<MemoryRouter initialEntries={['/canvas']}><NotificationCenter /></MemoryRouter>);
    const bell = screen.getByRole('button', { name: /Notifications/ });
    fireEvent.click(bell);
    fireEvent.click(screen.getByRole('button', { name: 'Review preview results' }));
    expect(useViewStore.getState().activeView).toBe('canvas');
    expect(useViewStore.getState().isResultsPanelExpanded).toBe(true);
    expect(bell).toHaveFocus();
  });

  it.each(['experiments', 'inference'] as const)('switches the internal %s view to canvas for submitted jobs', (activeView) => {
    // Job history must open over the intended canvas even when another canvas tab is active.
    const run = { label: 'Experiments', jobIds: ['job-a'] };
    useNotificationsStore.getState().upsertExecution('experiments:job-a', 'Submitted', { type: 'jobs', run });
    useViewStore.setState({ activeView });
    useJobStore.setState({ jobs: [], runJobs: {}, fetchJobs: vi.fn(), inspectedRun: null, isDrawerOpen: false });
    render(<MemoryRouter initialEntries={['/canvas']}><NotificationCenter /></MemoryRouter>);
    fireEvent.click(screen.getByRole('button', { name: /Notifications/ }));
    fireEvent.click(screen.getByRole('button', { name: 'View jobs' }));
    expect(useViewStore.getState().activeView).toBe('canvas');
    expect(useJobStore.getState().isDrawerOpen).toBe(true);
    expect(useJobStore.getState().inspectedRun).toEqual(run);
  });

  it('opens preview results from the bell without an error detail modal', () => {
    // Preview failures must lead directly to the canvas result that explains the problem.
    useNotificationsStore.getState().upsertExecution('canvas-preview', 'Preview blocked', { type: 'preview' }, 'warning');
    useViewStore.setState({ isResultsPanelExpanded: false });
    render(<MemoryRouter initialEntries={['/experiments']}><NotificationCenter /><LocationProbe /></MemoryRouter>);
    expect(screen.queryByRole('button', { name: 'Review preview results' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /Notifications/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Review preview results' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/canvas');
    expect(useViewStore.getState().isResultsPanelExpanded).toBe(true);
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Clear all' })).toBeNull();
  });

  it('renders live job statuses and opens the submitted group from another page', () => {
    // Completed runs must update inside the bell, and its action must preserve the exact job IDs.
    const run = { label: 'Experiments', jobIds: ['job-a'] };
    const makeJob = (status: JobInfo['status']): JobInfo => ({ job_id: 'job-a', pipeline_id: 'run', node_id: 'a', job_type: 'training', status,
      created_at: '2026-09-07T12:00:00Z', start_time: null, end_time: null, error: null, result: null });
    useNotificationsStore.getState().upsertExecution('experiments:job-a', 'Experiments submitted', { type: 'jobs', run });
    useJobStore.setState({ jobs: [makeJob('running')], runJobs: {}, fetchJobs: vi.fn(), inspectedRun: null, isDrawerOpen: false });
    const { container } = render(<MemoryRouter initialEntries={['/experiments']}><NotificationCenter /><LocationProbe /></MemoryRouter>);
    fireEvent.click(screen.getByRole('button', { name: /Notifications/ }));
    expect(screen.getByRole('status')).toHaveTextContent('Experiments: 1 running');
    act(() => useJobStore.setState({ jobs: [makeJob('completed')] }));
    expect(screen.getByRole('status')).toHaveTextContent('Experiments: 1 completed');
    expect(container.querySelector('button button')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'View jobs' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/canvas');
    expect(useJobStore.getState().isDrawerOpen).toBe(true);
    expect(useJobStore.getState().inspectedRun).toEqual(run);
    expect(screen.queryByRole('button', { name: 'Clear all' })).toBeNull();
  });
});

afterEach(() => {
  act(() => {
    useNotificationsStore.getState().clear();
    useJobStore.setState({ fetchJobs: originalFetchJobs, isDrawerOpen: false, inspectedRun: null });
    useViewStore.setState({ activeView: 'canvas' });
  });
  Element.prototype.getBoundingClientRect = originalRect;
});

describe('NotificationCenter detail modal', () => {
  it.each([
    { node_id: null, node_type: null, hasId: false, hasType: false },
    { node_id: 'node-zero', node_type: null, hasId: true, hasType: false },
    { node_id: '', node_type: 'Scaler', hasId: false, hasType: true },
    { node_id: 'node-one', node_type: 'Scaler', hasId: true, hasType: true },
  ])('preserves optional node detail fields for $node_id / $node_type', ({ node_id, node_type, hasId, hasType }) => {
    // Notifications with partial provenance must expose only the metadata the server supplied.
    useNotificationsStore.getState().clear();
    useNotificationsStore.getState().addMany([{ node_id, node_type, level: 'warning', logger: 'pipeline', message: 'Metadata warning' }]);
    render(<MemoryRouter><NotificationCenter /><LocationProbe /></MemoryRouter>);
    const bell = screen.getByRole('button', { name: 'Notifications (1)' });
    fireEvent.click(bell);
    expect(useNotificationsStore.getState().items[0]?.read).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: /Metadata warning/ }));
    const modal = screen.getByRole('dialog', { name: `${node_type ?? 'Pipeline notification'} details` });
    expect(within(modal).queryByText('Node ID') !== null).toBe(hasId);
    expect(within(modal).queryByText('Node type') !== null).toBe(hasType);
    expect(within(modal).getByText('Metadata warning')).toBeInTheDocument();
    fireEvent.click(within(modal).getByRole('button', { name: 'View in Error Log' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/errors');
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('focuses the modal on open, traps Tab, and returns focus to the bell button', async () => {
    render(
      <MemoryRouter>
        <NotificationCenter />
      </MemoryRouter>,
    );

    const bell = screen.getByRole('button', { name: /Notifications/ });
    bell.focus();
    fireEvent.click(bell);

    const row = await screen.findByRole('button', { name: /Training took too long/ });
    fireEvent.click(row);

    const closeButton = await screen.findByRole('button', { name: 'Close detail' });
    await waitFor(() => expect(closeButton).toHaveFocus());

    const modal = screen.getByRole('dialog', { name: /TrainingNode details/ });
    const modalButtons = within(modal).getAllByRole('button');
    const lastButton = modalButtons[modalButtons.length - 1]!;
    lastButton.focus();
    const tabEvent = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    window.dispatchEvent(tabEvent);
    expect(tabEvent.defaultPrevented).toBe(true);

    fireEvent.keyDown(document, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /TrainingNode details/ })).toBeNull());
    await waitFor(() => expect(bell).toHaveFocus());
  });

  it('keeps the row and Dismiss controls separate and independent', async () => {
    const user = userEvent.setup();

    const { container } = render(
      <MemoryRouter>
        <NotificationCenter />
      </MemoryRouter>,
    );

    const bell = screen.getByRole('button', { name: /Notifications/ });
    await act(async () => {
      await user.click(bell);
    });

    const row = await screen.findByRole('button', { name: /Training took too long/ });
    expect(container.querySelector('button button')).toBeNull();

    await act(async () => {
      await user.click(row);
    });
    await screen.findByRole('dialog', { name: /TrainingNode details/ });
    await act(async () => {
      await user.keyboard('{Escape}');
    });
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /TrainingNode details/ })).toBeNull());

    await act(async () => {
      await user.click(bell);
    });
    const dismiss = await screen.findByRole('button', { name: 'Dismiss' });
    await act(async () => {
      await user.click(dismiss);
    });

    await waitFor(() => expect(screen.queryByText('Training took too long')).toBeNull());
    expect(screen.queryByRole('dialog', { name: /TrainingNode details/ })).toBeNull();
  });
});
