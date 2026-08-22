import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Dashboard } from './Dashboard';

const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  getJobs: vi.fn(),
}));

vi.mock('../core/api/client', () => ({
  apiClient: { get: mocks.get },
}));

vi.mock('../core/api/jobs', () => ({
  jobsApi: { getJobs: mocks.getJobs },
}));

const renderDashboard = () => render(
  <MemoryRouter>
    <Dashboard />
  </MemoryRouter>,
);

describe('Dashboard', () => {
  beforeEach(() => {
    mocks.get.mockReset();
    mocks.getJobs.mockReset();
  });

  it('shows a getting-started guide instead of zeros on an empty workspace', async () => {
    mocks.get.mockResolvedValue({
      data: { total_jobs: 0, active_deployments: 0, data_sources: 0, training_jobs: 0, tuning_jobs: 0 },
    });
    mocks.getJobs.mockResolvedValue([]);

    renderDashboard();

    expect(await screen.findByText(/build your first pipeline/i)).toBeDefined();
    expect(screen.getByRole('link', { name: /add a dataset/i })).toBeDefined();
    expect(screen.getByRole('link', { name: /start from a template/i })).toBeDefined();
    expect(screen.queryByText('Recent Jobs')).toBeNull();
    expect(screen.queryByText('Weekly Activity')).toBeNull();
  });

  it('keeps the normal stats view once there is activity', async () => {
    mocks.get.mockResolvedValue({
      data: { total_jobs: 3, active_deployments: 0, data_sources: 1, training_jobs: 3, tuning_jobs: 0 },
    });
    mocks.getJobs.mockResolvedValue([
      {
        job_id: 'job-1',
        status: 'succeeded',
        model_type: 'RandomForest',
        start_time: '2026-08-20T10:00:00.000Z',
      },
    ]);

    renderDashboard();

    expect(await screen.findByText('Recent Jobs')).toBeDefined();
    expect(screen.queryByText(/build your first pipeline/i)).toBeNull();
  });
});
