import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import { DeploymentsPage } from './DeploymentsPage';
import { ConfirmProvider } from '../shared';
import { deploymentApi, DeploymentInfo } from '../../core/api/deployment';
import { serializeOperationalContext } from '../../core/utils/operationalContext';

vi.mock('../../core/api/deployment', async () => {
  const actual = await vi.importActual<typeof import('../../core/api/deployment')>('../../core/api/deployment');
  return {
    ...actual,
    deploymentApi: {
      deployModel: vi.fn(),
      getActive: vi.fn(),
      getHistory: vi.fn(),
      deactivate: vi.fn(),
      predict: vi.fn(),
    },
  };
});

const makeDeployment = (overrides: Partial<DeploymentInfo> & Pick<DeploymentInfo, 'id' | 'job_id'>): DeploymentInfo => ({
  model_type: 'random_forest',
  artifact_uri: 's3://bucket/model.pkl',
  is_active: false,
  created_at: '2026-01-01T00:00:00Z',
  version: 1,
  dataset_id: 'ds-1',
  previous_deployment_id: null,
  ...overrides,
});

function renderDeployments(initialEntry = '/deployments') {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <ConfirmProvider>
        <DeploymentsPage />
      </ConfirmProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.mocked(deploymentApi.getActive).mockReset();
  vi.mocked(deploymentApi.getHistory).mockReset();
  vi.mocked(deploymentApi.deactivate).mockReset();
  vi.mocked(deploymentApi.deployModel).mockReset();
});

// jsdom does not implement scrollIntoView; the deep-link highlight effect calls it.
Element.prototype.scrollIntoView = vi.fn();

describe('DeploymentsPage lineage (OPS-002)', () => {
  it('renders the active deployment lineage as RecordLinks to job, version and dataset', async () => {
    const active = makeDeployment({ id: 1, job_id: 'job-active', is_active: true, version: 4, dataset_id: 'ds-9' });
    vi.mocked(deploymentApi.getActive).mockResolvedValue(active);
    vi.mocked(deploymentApi.getHistory).mockResolvedValue([active]);

    renderDeployments();

    expect((await screen.findAllByText('random_forest')).length).toBeGreaterThan(0);
    const [jobLink] = screen.getAllByRole('link', { name: /Job job-active/i });
    expect(jobLink?.getAttribute('href')?.split('?')[0]).toBe('/jobs');
    const [versionLink] = screen.getAllByRole('link', { name: /version 4/i });
    expect(versionLink?.getAttribute('href')?.split('?')[0]).toBe('/registry');
    const [datasetLink] = screen.getAllByRole('link', { name: /dataset/i });
    expect(datasetLink?.getAttribute('href')?.split('?')[0]).toBe('/data');
  });

  it('states "no target available" / "no prior deployment" when lineage data is missing', async () => {
    const active = makeDeployment({
      id: 2,
      job_id: 'job-unknown',
      is_active: true,
      version: null,
      dataset_id: 'unknown',
      previous_deployment_id: null,
    });
    vi.mocked(deploymentApi.getActive).mockResolvedValue(active);
    vi.mocked(deploymentApi.getHistory).mockResolvedValue([active]);

    renderDeployments();

    expect((await screen.findAllByText('random_forest')).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/version unavailable/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/no dataset available/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/no prior deployment/i).length).toBeGreaterThan(0);
  });

  it('links a redeployed deployment to the one it replaced', async () => {
    const active = makeDeployment({
      id: 3,
      job_id: 'job-3',
      is_active: true,
      version: 2,
      previous_deployment_id: 1,
    });
    vi.mocked(deploymentApi.getActive).mockResolvedValue(active);
    vi.mocked(deploymentApi.getHistory).mockResolvedValue([active]);

    renderDeployments();

    const replacedLinks = await screen.findAllByRole('link', { name: /deployment 1/i });
    expect(replacedLinks[0]?.getAttribute('href')?.split('?')[0]).toBe('/deployments');
  });

  it('names the exact deployment in the deactivate confirmation and prevents double submit', async () => {
    const user = userEvent.setup();
    const active = makeDeployment({ id: 5, job_id: 'job-5', is_active: true, version: 3 });
    vi.mocked(deploymentApi.getActive).mockResolvedValue(active);
    vi.mocked(deploymentApi.getHistory).mockResolvedValue([active]);
    vi.mocked(deploymentApi.deactivate).mockImplementation(
      () => new Promise((resolve) => setTimeout(resolve, 20)),
    );

    renderDeployments();
    await screen.findAllByText('random_forest');

    await user.click(screen.getByRole('button', { name: /deactivate/i }));
    expect(await screen.findByText(/deactivate random_forest v3 \(job job-5\)/i)).toBeInTheDocument();
    const confirmDialog = screen.getByRole('dialog');
    await user.click(within(confirmDialog).getByRole('button', { name: /^deactivate$/i }));

    // The button must disable while the mutation is in flight so a second
    // click cannot fire a duplicate deactivate request.
    const deactivateButton = await screen.findByRole('button', { name: /deactivate/i });
    expect(deactivateButton).toBeDisabled();

    await waitFor(() => expect(deploymentApi.deactivate).toHaveBeenCalledTimes(1));
  });

  it('scopes redeploy failure to the affected job and retries in place without re-confirming', async () => {
    const user = userEvent.setup();
    const active = makeDeployment({ id: 10, job_id: 'job-active', is_active: true, version: 1 });
    const inactive = makeDeployment({ id: 11, job_id: 'job-inactive', is_active: false, version: 2 });
    vi.mocked(deploymentApi.getActive).mockResolvedValue(active);
    vi.mocked(deploymentApi.getHistory).mockResolvedValue([active, inactive]);
    vi.mocked(deploymentApi.deployModel)
      .mockRejectedValueOnce(new Error('deploy failed'))
      .mockResolvedValueOnce(active);

    renderDeployments();
    await screen.findAllByText('random_forest');

    await user.click(screen.getByRole('button', { name: /redeploy random_forest v2 \(job job-inactive\)/i }));
    const redeployDialog = await screen.findByRole('dialog');
    await user.click(within(redeployDialog).getByRole('button', { name: /^redeploy$/i }));

    expect(await screen.findByText(/failed to redeploy job job-inactive/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /retry/i }));

    await waitFor(() => expect(deploymentApi.deployModel).toHaveBeenCalledTimes(2));
    expect(deploymentApi.deployModel).toHaveBeenNthCalledWith(2, 'job-inactive');
  });

  it('highlights the deployment named in a deep-link context on load', async () => {
    const active = makeDeployment({ id: 20, job_id: 'job-a', is_active: true, version: 1 });
    const target = makeDeployment({ id: 21, job_id: 'job-b', is_active: false, version: 2 });
    vi.mocked(deploymentApi.getActive).mockResolvedValue(active);
    vi.mocked(deploymentApi.getHistory).mockResolvedValue([active, target]);

    const query = serializeOperationalContext({ ref: { kind: 'deployment', deploymentId: 21 } });
    renderDeployments(`/deployments${query}`);

    await screen.findAllByText('random_forest');
    const row = await screen.findByText('job-b', { exact: false });
    expect(row.closest('tr')).toHaveClass('ring-1');
  });
});
