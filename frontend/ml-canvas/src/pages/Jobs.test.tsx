import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { JobsPage } from './Jobs';
import { jobsApi, JobInfo } from '../core/api/jobs';
import { registryApi } from '../core/api/registry';

// JobDetailsView pulls in useJobPolling/useJobStore/log rendering — none of
// that is under test here. Jobs.tsx's own responsibilities (Details link
// wiring, URL-carried list state, fallback lookup, restore-on-back) are
// isolated behind a lightweight stub that exposes just what these tests
// assert on.
vi.mock('../components/panels/jobs/JobDetailsView', () => ({
  JobDetailsView: ({ job, onBack }: { job: JobInfo; onBack: () => void }) => (
    <div>
      <p>Details for {job.job_id}</p>
      <button onClick={onBack}>Back</button>
    </div>
  ),
}));

vi.mock('../core/api/registry', () => ({
  registryApi: { getAllNodes: vi.fn() },
}));

vi.mock('../core/api/jobs', async () => {
  const actual = await vi.importActual<typeof import('../core/api/jobs')>('../core/api/jobs');
  return {
    ...actual,
    jobsApi: {
      ...actual.jobsApi,
      getJobs: vi.fn(),
      getEDAJobs: vi.fn(),
      getIngestionJobs: vi.fn(),
      getJob: vi.fn(),
    },
  };
});

const makeJob = (overrides: Partial<JobInfo> & Pick<JobInfo, 'job_id'>): JobInfo => ({
  pipeline_id: 'pipe-1',
  node_id: 'node-1',
  job_type: 'training',
  status: 'completed',
  start_time: '2026-01-01T00:00:00Z',
  end_time: '2026-01-01T00:05:00Z',
  error: null,
  result: null,
  model_type: 'random_forest',
  created_at: '2026-01-01T00:00:00Z',
  ...overrides,
});

function renderJobsPage(initialEntry = '/jobs') {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/jobs" element={<JobsPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('JobsPage', () => {
  beforeEach(() => {
    vi.mocked(registryApi.getAllNodes).mockResolvedValue([
      { id: 'random_forest', name: 'Random Forest', category: 'model', description: '', tags: ['classification'], params: {} },
    ]);
    vi.mocked(jobsApi.getEDAJobs).mockResolvedValue([]);
    vi.mocked(jobsApi.getIngestionJobs).mockResolvedValue([]);
  });

  it('renders a paginated multi-status job pool with a Details link per row', async () => {
    vi.mocked(jobsApi.getJobs).mockResolvedValue([
      makeJob({ job_id: 'job-completed', status: 'completed' }),
      makeJob({ job_id: 'job-failed', status: 'failed' }),
      makeJob({ job_id: 'job-running', status: 'running' }),
    ]);

    renderJobsPage();

    await waitFor(() => {
      expect(screen.getAllByText('View details')).toHaveLength(3);
    });
  });

  it('shows the real model family + run mode instead of the raw job_type ("training"/"tuning")', async () => {
    vi.mocked(jobsApi.getJobs).mockResolvedValue([
      makeJob({ job_id: 'job-basic', job_type: 'training', model_type: 'random_forest' }),
    ]);

    renderJobsPage();

    expect(await screen.findByText('Classification (basic)')).toBeInTheDocument();
    expect(screen.queryByText('training')).not.toBeInTheDocument();
  });

  it('opens the Details view for the clicked job and returns to the list on Back', async () => {
    const user = userEvent.setup();
    vi.mocked(jobsApi.getJobs).mockResolvedValue([
      makeJob({ job_id: 'job-1' }),
      makeJob({ job_id: 'job-2' }),
    ]);

    renderJobsPage();

    const links = await screen.findAllByText('View details');
    await act(async () => {
      await user.click(links[0]!);
    });

    expect(await screen.findByText('Details for job-1')).toBeInTheDocument();

    await act(async () => {
      await user.click(screen.getByText('Back'));
    });

    // Back navigates to the plain /jobs URL and the list re-renders.
    expect(await screen.findAllByText('View details')).toHaveLength(2);
  });

  it('restores search/status filters carried on the Details link when the page reloads mid-investigation', async () => {
    vi.mocked(jobsApi.getJobs).mockResolvedValue([makeJob({ job_id: 'job-1', status: 'failed' })]);
    vi.mocked(jobsApi.getJob).mockResolvedValue(makeJob({ job_id: 'job-1', status: 'failed' }));

    // Simulates a deep link / reload while Details is open: the query
    // string carries the exact filters the user had applied before opening it.
    renderJobsPage('/jobs?oc.kind=job&oc.jobId=job-1&oc.origin=%2Fjobs&oc.f.tab=classification&oc.f.q=job&oc.f.status=failed');

    expect(await screen.findByText('Details for job-1')).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByText('Back'));
    });

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search jobs...')).toHaveValue('job');
    });
    // The status filter panel is collapsed by default; open it to confirm
    // the restored value rather than assuming its initial visibility.
    fireEvent.click(screen.getByRole('button', { name: /Filters/i }));
    expect(screen.getByRole('combobox')).toHaveValue('failed');
  });

  it('fetches the job via the API as a fallback when it is not present in the loaded pool', async () => {
    vi.mocked(jobsApi.getJobs).mockResolvedValue([makeJob({ job_id: 'job-other' })]);
    vi.mocked(jobsApi.getJob).mockResolvedValue(makeJob({ job_id: 'job-not-in-pool' }));

    renderJobsPage('/jobs?oc.kind=job&oc.jobId=job-not-in-pool');

    await waitFor(() => {
      expect(jobsApi.getJob).toHaveBeenCalledWith('job-not-in-pool');
    });
    expect(await screen.findByText('Details for job-not-in-pool')).toBeInTheDocument();
  });

  it('shows an explanatory error (not a blank/broken screen) when the linked job cannot be found', async () => {
    vi.mocked(jobsApi.getJobs).mockResolvedValue([]);
    vi.mocked(jobsApi.getJob).mockRejectedValue(new Error('404'));

    renderJobsPage('/jobs?oc.kind=job&oc.jobId=missing-job');

    expect(await screen.findByRole('alert')).toHaveTextContent(/could not be found/i);
    expect(screen.getByText('Back to Jobs')).toBeInTheDocument();
  });
});
