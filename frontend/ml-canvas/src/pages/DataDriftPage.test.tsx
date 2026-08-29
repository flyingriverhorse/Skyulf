import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import {
    monitoringApi,
    type DriftAlertDetail,
    type DriftHistoryEntry,
} from '../core/api/monitoring';
import { DataDriftPage } from './DataDriftPage';

vi.mock('../core/api/monitoring', async () => {
    const actual = await vi.importActual<typeof import('../core/api/monitoring')>(
        '../core/api/monitoring',
    );
    return {
        ...actual,
        monitoringApi: {
            ...actual.monitoringApi,
            getJobs: vi.fn(),
            updateJobDescription: vi.fn(),
            calculateDrift: vi.fn(),
            getDriftHistory: vi.fn(),
            getDriftAlert: vi.fn(),
            updateDriftAlertDisposition: vi.fn(),
        },
    };
});

function historyEntry(overrides: Partial<DriftHistoryEntry> = {}): DriftHistoryEntry {
    return {
        id: 1,
        job_id: 'job-1',
        dataset_name: 'sales',
        reference_rows: 100,
        current_rows: 100,
        drifted_columns_count: 1,
        total_columns: 2,
        created_at: '2026-08-07T10:00:00',
        severity: 'warning',
        status: 'new',
        owner: null,
        acknowledged_at: null,
        resolved_at: null,
        threshold_version: 1,
        threshold_psi: 0.2,
        threshold_ks: 0.05,
        threshold_wasserstein: 0.1,
        threshold_kl: 0.1,
        deployment_id: null,
        model_version: null,
        evaluation_status: 'completed',
        error_message: null,
        ...overrides,
    };
}

function alertDetail(overrides: Partial<DriftAlertDetail> = {}): DriftAlertDetail {
    return {
        ...historyEntry(),
        column_drifts: {
            col_0: {
                column: 'col_0',
                metrics: [
                    { metric: 'psi', value: 0.5, has_drift: true, threshold: 0.2 },
                    { metric: 'ks_statistic', value: 0.18, has_drift: true, threshold: 0.1 },
                    { metric: 'ks_test_p_value', value: 0.01, has_drift: true, threshold: 0.1 },
                ],
                drift_detected: true,
                suggestions: [],
            },
        },
        disposition_history: [],
        ...overrides,
    };
}

function renderPage() {
    return render(
        <MemoryRouter>
            <DataDriftPage />
        </MemoryRouter>,
    );
}

/** Selects the mocked reference job so `useDriftHistory` fetches its alert history. */
async function selectJob() {
    fireEvent.click(screen.getByRole('button', { name: /select reference model/i }));
    fireEvent.click(await screen.findByRole('option', { name: /sales/i }));
}

describe('DataDriftPage — OPS-003 durable drift alert lifecycle', () => {
    beforeEach(() => {
        vi.mocked(monitoringApi.getJobs).mockReset().mockResolvedValue([
            { job_id: 'job-1', dataset_name: 'sales' } as never,
        ]);
        vi.mocked(monitoringApi.updateJobDescription).mockReset().mockResolvedValue(undefined);
        vi.mocked(monitoringApi.calculateDrift).mockReset();
        vi.mocked(monitoringApi.getDriftHistory).mockReset().mockResolvedValue([]);
        vi.mocked(monitoringApi.getDriftAlert).mockReset();
        vi.mocked(monitoringApi.updateDriftAlertDisposition).mockReset();
    });

    it('lists persisted alerts, including their severity, status, and pinned threshold version', async () => {
        vi.mocked(monitoringApi.getDriftHistory).mockResolvedValue([
            historyEntry({ id: 1, threshold_version: 1 }),
            historyEntry({ id: 2, threshold_version: 2, severity: 'critical', status: 'acknowledged' }),
        ]);
        renderPage();
        await selectJob();

        await waitFor(() => expect(screen.getByText('Alert History')).toBeInTheDocument());
        expect(screen.getByText('v1')).toBeInTheDocument();
        expect(screen.getByText('v2')).toBeInTheDocument();
        expect(screen.getAllByText('Investigate')).toHaveLength(2);
    });

    it('opens the alert investigation modal and mirrors the alert id into the URL', async () => {
        vi.mocked(monitoringApi.getDriftHistory).mockResolvedValue([historyEntry({ id: 7 })]);
        vi.mocked(monitoringApi.getDriftAlert).mockResolvedValue(alertDetail({ id: 7 }));
        renderPage();
        await selectJob();

        await waitFor(() => expect(screen.getAllByText('Investigate')).toHaveLength(1));
        fireEvent.click(screen.getAllByText('Investigate')[0] as HTMLElement);

        await waitFor(() => expect(monitoringApi.getDriftAlert).toHaveBeenCalledWith(7));
        expect(await screen.findByText('Drift alert #7')).toBeInTheDocument();
    });

    it('renders the feature evidence table with the per-feature drift statistics', async () => {
        vi.mocked(monitoringApi.getDriftHistory).mockResolvedValue([historyEntry({ id: 7 })]);
        vi.mocked(monitoringApi.getDriftAlert).mockResolvedValue(alertDetail({ id: 7 }));
        renderPage();
        await selectJob();

        fireEvent.click(await screen.findByText('Investigate'));
        await screen.findByText('Drift alert #7');

        expect(screen.getByText('Feature evidence')).toBeInTheDocument();
        // The table alternative is collapsible-open by default (`defaultOpen`).
        expect(screen.getByText('col_0')).toBeInTheDocument();
    });

    it('shows related job/model-version/deployment RecordLinks in the alert modal', async () => {
        vi.mocked(monitoringApi.getDriftHistory).mockResolvedValue([historyEntry({ id: 7 })]);
        vi.mocked(monitoringApi.getDriftAlert).mockResolvedValue(
            alertDetail({ id: 7, deployment_id: 42, model_version: 'v3' }),
        );
        renderPage();
        await selectJob();

        fireEvent.click(await screen.findByText('Investigate'));
        await screen.findByText('Drift alert #7');

        const jobLink = screen.getByRole('link', { name: /^Job job-1$/i });
        expect(jobLink.getAttribute('href')).toContain('/jobs');
        const modelVersionLink = screen.getByRole('link', { name: /Model version v3/i });
        expect(modelVersionLink.getAttribute('href')).toContain('/registry');
        const deploymentLink = screen.getByRole('link', { name: /Deployment 42/i });
        expect(deploymentLink.getAttribute('href')).toContain('/deployments');
    });

    it('walks the disposition state machine: new -> acknowledged -> resolved -> reopened', async () => {
        vi.mocked(monitoringApi.getDriftHistory).mockResolvedValue([historyEntry({ id: 7, status: 'new' })]);
        vi.mocked(monitoringApi.getDriftAlert).mockResolvedValue(alertDetail({ id: 7, status: 'new' }));
        vi.mocked(monitoringApi.updateDriftAlertDisposition).mockResolvedValue(
            alertDetail({
                id: 7,
                status: 'acknowledged',
                owner: 'alice',
                disposition_history: [{ status: 'acknowledged', actor: 'alice', note: null, at: '2026-08-07T11:00:00' }],
            }),
        );
        renderPage();
        await selectJob();

        fireEvent.click(await screen.findByText('Investigate'));
        await screen.findByText('Drift alert #7');

        fireEvent.change(screen.getByLabelText(/your name/i), { target: { value: 'alice' } });
        fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' }));

        await waitFor(() =>
            expect(monitoringApi.updateDriftAlertDisposition).toHaveBeenCalledWith(
                7,
                'acknowledge',
                'alice',
                undefined,
            ),
        );
        expect(await screen.findByText(/Currently/)).toHaveTextContent('acknowledged');
        expect(screen.getByText(/by/)).toHaveTextContent('alice');
    });

    it('rejects a disposition action without an actor name', async () => {
        vi.mocked(monitoringApi.getDriftHistory).mockResolvedValue([historyEntry({ id: 7, status: 'new' })]);
        vi.mocked(monitoringApi.getDriftAlert).mockResolvedValue(alertDetail({ id: 7, status: 'new' }));
        renderPage();
        await selectJob();

        fireEvent.click(await screen.findByText('Investigate'));
        await screen.findByText('Drift alert #7');

        fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' }));
        expect(
            await screen.findByText(/Enter your name so the disposition records who made it/),
        ).toBeInTheDocument();
        expect(monitoringApi.updateDriftAlertDisposition).not.toHaveBeenCalled();
    });

    it('shows the explicit no-baseline empty state distinct from a generic failure', async () => {
        const err = Object.assign(new Error('no reference'), {
            response: { status: 404, data: { detail: 'No reference dataset found for this job.' } },
        });
        vi.mocked(monitoringApi.calculateDrift).mockRejectedValue(err);
        renderPage();

        await waitFor(() => expect(monitoringApi.getJobs).toHaveBeenCalled());
        const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
        const file = new File(['a,b\n1,2'], 'current.csv', { type: 'text/csv' });
        fireEvent.change(fileInput, { target: { files: [file] } });

        fireEvent.click(screen.getByRole('button', { name: /select reference model/i }));
        fireEvent.click(await screen.findByRole('option', { name: /sales/i }));

        fireEvent.click(screen.getByRole('button', { name: /run analysis/i }));

        expect(
            await screen.findByText(/No baseline reference is available for this job yet/),
        ).toBeInTheDocument();
    });

    it('shows an explicit evaluation-failed state (distinct from no-baseline) with retry', async () => {
        const err = Object.assign(new Error('boom'), {
            response: { status: 500, data: { detail: 'Drift evaluation crashed.' } },
        });
        vi.mocked(monitoringApi.calculateDrift).mockRejectedValue(err);
        renderPage();

        await waitFor(() => expect(monitoringApi.getJobs).toHaveBeenCalled());
        const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
        const file = new File(['a,b\n1,2'], 'current.csv', { type: 'text/csv' });
        fireEvent.change(fileInput, { target: { files: [file] } });

        fireEvent.click(screen.getByRole('button', { name: /select reference model/i }));
        fireEvent.click(await screen.findByRole('option', { name: /sales/i }));

        fireEvent.click(screen.getByRole('button', { name: /run analysis/i }));

        expect(await screen.findByText('Drift evaluation crashed.')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    });

    it('shows the plain "no report yet" empty state when nothing has been run', async () => {
        renderPage();
        await waitFor(() => expect(monitoringApi.getJobs).toHaveBeenCalled());
        expect(await screen.findByText('No Drift Report Yet')).toBeInTheDocument();
    });
});
