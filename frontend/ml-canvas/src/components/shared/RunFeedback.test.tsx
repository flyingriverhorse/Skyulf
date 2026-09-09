import { act, fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import type { JobInfo } from '../../core/api/jobs';
import { useJobStore } from '../../core/store/useJobStore';
import { RunFeedback, summariseRunJobs } from './RunFeedback';

/** Keep job transitions realistic without starting backend polling. */
function job(job_id: string, status: JobInfo['status']): JobInfo {
  return { job_id, status, pipeline_id: 'run', node_id: 'model', job_type: 'training',
    created_at: '2026-09-07T12:00:00Z', start_time: null, end_time: null, error: null, result: null };
}

beforeEach(() => useJobStore.setState({ jobs: [], runJobs: {}, toggleDrawer: vi.fn() }));

it('follows only the submitted jobs and opens their existing jobs drawer', () => {
  // Historical jobs must not turn a fresh submission into a completed run.
  useJobStore.setState({ jobs: [job('old', 'completed')] });
  render(<RunFeedback run={{ label: 'Training — Random forest', jobIds: ['new'] }} />);
  expect(screen.getByRole('status')).toHaveTextContent('Training — Random forest: Awaiting status');
  act(() => useJobStore.setState({ jobs: [job('new', 'queued')] }));
  expect(screen.getByRole('status')).toHaveTextContent('Training — Random forest: 1 queued');
  act(() => useJobStore.setState({ jobs: [job('new', 'running')] }));
  expect(screen.getByRole('status')).toHaveTextContent('1 running');
  act(() => useJobStore.setState({ jobs: [job('new', 'succeeded')] }));
  expect(screen.getByRole('status')).toHaveTextContent('1 completed');
  fireEvent.click(screen.getByRole('button', { name: 'View jobs' }));
  expect(useJobStore.getState().toggleDrawer).toHaveBeenCalledWith(true);
  expect(useJobStore.getState().inspectedRun?.jobIds).toEqual(['new']);
});

it('keeps failures, cancellations and missing branch statuses visible together', () => {
  // A partial success must never announce that every experiment completed.
  useJobStore.setState({ jobs: [job('a', 'completed'), job('b', 'failed'), job('c', 'cancelled'), job('d', 'pending')] });
  render(<RunFeedback run={{ label: 'Experiments', jobIds: ['a', 'b', 'c', 'd', 'missing'] }} />);
  const status = screen.getByRole('status');
  expect(status).toHaveTextContent('1 completed');
  expect(status).toHaveTextContent('1 failed');
  expect(status).toHaveTextContent('1 cancelled');
  expect(status).toHaveTextContent('1 pending');
  expect(status).toHaveTextContent('1 awaiting status');
});

it('opens the model tab without grouping when feedback belongs to a node', () => {
  // Reopening node history must not reuse an earlier Run all scope.
  useJobStore.setState({ inspectedRun: { label: 'Experiments', jobIds: ['older'] }, setTab: vi.fn() });
  render(<RunFeedback run={{ label: 'Training', jobIds: ['new'] }} task="regression" />);
  fireEvent.click(screen.getByRole('button', { name: 'View jobs' }));
  expect(useJobStore.getState().setTab).toHaveBeenCalledWith('regression');
  expect(useJobStore.getState().toggleDrawer).toHaveBeenCalledWith(true);
  expect(useJobStore.getState().inspectedRun).toBeNull();
});

it('summarises only the submitted job ids', () => {
  const summary = summariseRunJobs(
    ['new', 'missing'],
    [job('old', 'completed'), job('new', 'queued')],
    {},
  );

  expect(summary).toBe('1 queued · 1 awaiting status');
});
