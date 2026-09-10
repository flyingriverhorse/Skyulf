import type { ComponentProps } from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import type { DriftAlertDetail, DriftAlertStatus } from '../../core/api/monitoring';
import { DriftAlertModal } from './DriftAlertModal';

/** Supply a complete alert while keeping optional metadata absent by default. */
function detail(overrides: Partial<DriftAlertDetail> = {}): DriftAlertDetail {
    return {
        id: 7, job_id: 'job-1', severity: 'warning', status: 'new',
        evaluation_status: 'completed', disposition_history: [], ...overrides,
    };
}

/** Exercise the public modal with its real shell, fields, table and contextual links. */
function setup(overrides: Partial<ComponentProps<typeof DriftAlertModal>> = {}) {
    let props: ComponentProps<typeof DriftAlertModal> = {
        alertId: 7, detail: detail(), loading: false, error: null, actionPending: false,
        onApplyDisposition: vi.fn().mockResolvedValue(null), onRetry: vi.fn(), onClose: vi.fn(),
        filters: { severity: 'warning', job: 'job-1' }, ...overrides,
    };
    const view = render(<MemoryRouter><DriftAlertModal {...props} /></MemoryRouter>);
    return {
        props,
        rerender(patch: Partial<typeof props>) {
            props = { ...props, ...patch };
            view.rerender(<MemoryRouter><DriftAlertModal {...props} /></MemoryRouter>);
        },
    };
}

/** Fill the audit fields through their accessible labels. */
function fill(actor: string, note = '') {
    fireEvent.change(screen.getByRole('textbox', { name: /Your name/ }), { target: { value: actor } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Note' }), { target: { value: note } });
}

describe('DriftAlertModal characterization', () => {
    it('closes only for a null ID and retains the accessible zero-ID title', () => {
        /** Zero is a valid public alert ID and must not be treated as closed. */
        const view = setup({ alertId: null });
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        view.rerender({ alertId: 0 });
        expect(screen.getByRole('dialog', { name: 'Drift alert #0' })).toHaveAttribute('aria-modal', 'true');
    });

    it('gives loading precedence over both detail and error', () => {
        /** Refreshes temporarily unmount detail even when stale data exists. */
        setup({ loading: true, error: 'Network failed' });
        expect(screen.getByText('Loading drift alert…')).toBeInTheDocument();
        expect(screen.queryByText('Disposition')).not.toBeInTheDocument();
        expect(screen.queryByText('Network failed')).not.toBeInTheDocument();
    });

    it('shows a retryable error without detail, then an empty body without either', () => {
        /** Errors must remain recoverable while the no-result state stays empty. */
        const view = setup({ detail: null, error: 'Network failed' });
        fireEvent.click(screen.getByRole('button', { name: /retry/i }));
        expect(view.props.onRetry).toHaveBeenCalledOnce();
        view.rerender({ error: null });
        expect(screen.queryByText('Network failed')).not.toBeInTheDocument();
        expect(screen.queryByText('Disposition')).not.toBeInTheDocument();
    });

    it('keeps detail visible with the action error and supports close and Escape', () => {
        /** Existing detail takes precedence over the full-page error surface. */
        const view = setup({ error: 'Action failed' });
        expect(screen.getByText('Disposition')).toBeInTheDocument();
        expect(screen.getByText('Action failed')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /retry/i })).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Close' }));
        fireEvent.keyDown(window, { key: 'Escape' });
        expect(view.props.onClose).toHaveBeenCalledTimes(2);
    });

    it('preserves absent and empty metadata fallbacks and zero threshold/count values', () => {
        /** Nullish metadata and numeric zero intentionally have different displays. */
        const view = setup();
        expect(screen.getByText('Unknown')).toBeInTheDocument();
        expect(screen.getAllByText('—')).toHaveLength(2);
        view.rerender({ detail: detail({ created_at: '', threshold_version: 0, threshold_psi: 0,
            threshold_ks: 0.05, threshold_wasserstein: 0.1, threshold_kl: 0.2,
            drifted_columns_count: 0, total_columns: 0 }) });
        expect(screen.queryByText('Unknown')).not.toBeInTheDocument();
        expect(screen.getByText('v0')).toHaveTextContent('PSI 0 · KS 0.05 · Wasserstein 0.1 · KL 0.2');
        expect(screen.getByText('0 / 0')).toBeInTheDocument();
        view.rerender({ detail: detail({ created_at: '2026-09-10T12:34:56', drifted_columns_count: 1 }) });
        expect(screen.getByText('2026-09-10 12:34')).toBeInTheDocument();
        expect(screen.getAllByText('—')).toHaveLength(2);
    });

    it.each(['no_baseline', 'failed'] as const)('shows %s evaluation context without evidence', status => {
        /** Incomplete evaluations explain their outcome instead of presenting evidence. */
        const view = setup({ detail: detail({ evaluation_status: status, error_message: 'Evaluation context' }) });
        expect(screen.getByText(status === 'no_baseline' ? 'No baseline' : 'Evaluation failed')).toBeInTheDocument();
        expect(screen.getByText('Evaluation context')).toBeInTheDocument();
        expect(screen.queryByText('Feature evidence')).not.toBeInTheDocument();
        view.rerender({ detail: detail({ error_message: 'Evaluation context' }) });
        expect(screen.queryByText('Evaluation context')).not.toBeInTheDocument();
        expect(screen.getByText('No evidence recorded')).toBeInTheDocument();
    });

    it.each([undefined, null, {}])('uses the empty evidence surface for %s', column_drifts => {
        /** Missing and empty retained evidence share the same user explanation. */
        setup({ detail: detail(column_drifts === undefined ? {} : { column_drifts }) });
        expect(screen.getByText('No evidence recorded')).toBeInTheDocument();
        expect(screen.queryByRole('table')).not.toBeInTheDocument();
    });

    it('preserves feature/metric order, last duplicate values, rounding and missing cells', () => {
        /** Exported evidence must keep the original metric mapping and row order. */
        setup({ detail: detail({ column_drifts: {
            z: { column: 'ignored', drift_detected: true, suggestions: [], metrics: [
                { metric: 'psi', value: 9, has_drift: true, threshold: 1 },
                { metric: 'psi', value: 0.123456, has_drift: true, threshold: 1 },
                { metric: 'wasserstein_distance', value: 0, has_drift: false, threshold: 1 },
                { metric: 'ks_statistic', value: 0.99999, has_drift: true, threshold: 1 },
                { metric: 'ks_test_p_value', value: 0.00001, has_drift: true, threshold: 1 },
            ] },
            a: { column: 'a', drift_detected: false, suggestions: [], metrics: [] },
        } }) });
        const table = screen.getByRole('table', { name: 'Per-feature drift evidence for alert #7' });
        expect(within(table).getAllByRole('columnheader').map(cell => cell.textContent))
            .toEqual(['Feature', 'Drifted', 'PSI', 'Wasserstein', 'KS statistic', 'KS p-value']);
        expect(within(table).getAllByRole('row').slice(1).map(row => within(row).getAllByRole('cell').map(cell => cell.textContent)))
            .toEqual([['z', 'Yes', '0.1235', '0', '1', '0'], ['a', 'No', '—', '—', '—', '—']]);
        expect(screen.getByRole('button', { name: 'Hide data table' })).toHaveAttribute('aria-expanded', 'true');
    });

    it('preserves table collapse across alert changes and resets it after loading or closing', () => {
        /** Evidence keeps its existing mount lifetime independently of the audit field state. */
        const evidenceDetail = detail({ column_drifts: {
            feature: { column: 'feature', drift_detected: false, suggestions: [], metrics: [] },
        } });
        const view = setup({ detail: evidenceDetail });
        fireEvent.click(screen.getByRole('button', { name: 'Hide data table' }));
        view.rerender({ alertId: 8, detail: { ...evidenceDetail, id: 8 } });
        expect(screen.queryByRole('table')).not.toBeInTheDocument();
        view.rerender({ loading: true });
        view.rerender({ loading: false });
        expect(screen.getByRole('table')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Hide data table' }));
        view.rerender({ alertId: null });
        view.rerender({ alertId: 8 });
        expect(screen.getByRole('table')).toBeInTheDocument();
    });

    it('preserves job, model version and zero deployment links with originating filters', () => {
        /** Investigation navigation must retain the drift context for return links. */
        const view = setup({ detail: detail({ model_version: 'v3', deployment_id: 0 }) });
        const links = screen.getAllByRole('link');
        expect(links.map(link => link.getAttribute('aria-label'))).toEqual(['Job job-1', 'Model version v3 (job job-1)', 'Deployment 0']);
        for (const link of links) {
            expect(decodeURIComponent(link.getAttribute('href') ?? '')).toContain('/drift');
            expect(decodeURIComponent(link.getAttribute('href') ?? '')).toContain('warning');
        }
        view.rerender({ detail: detail({ model_version: '', deployment_id: null }) });
        expect(screen.getAllByRole('link')).toHaveLength(1);
    });

    it.each([
        ['new', ['Acknowledge']], ['acknowledged', ['Resolve', 'Reopen']],
        ['resolved', ['Reopen']], ['reopened', ['Acknowledge']],
    ] as const)('keeps ordered actions for %s', (status, expected) => {
        /** Each disposition exposes only its established next transitions. */
        setup({ detail: detail({ status: status as DriftAlertStatus }) });
        expect(screen.queryAllByRole('button').filter(button => ['Acknowledge', 'Resolve', 'Reopen'].includes(button.textContent ?? '')).map(button => button.textContent))
            .toEqual(expected);
        expect(screen.queryAllByRole('textbox')).toHaveLength(expected.length ? 2 : 0);
    });

    it('validates trimmed actor, preserves validation while typing and trims successful arguments', async () => {
        /** Audit actions cannot lose actor validation or alter callback argument normalization. */
        const onApplyDisposition = vi.fn().mockResolvedValue({ ok: true });
        setup({ onApplyDisposition });
        fill('   ', '  context  ');
        fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' }));
        expect(onApplyDisposition).not.toHaveBeenCalled();
        expect(screen.getByText(/Enter your name so/)).toBeInTheDocument();
        fill(' alice ', '  context  ');
        expect(screen.getByText(/Enter your name so/)).toBeInTheDocument();
        await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' })));
        expect(onApplyDisposition).toHaveBeenCalledWith('acknowledge', 'alice', 'context');
        expect(screen.queryByText(/Enter your name so/)).not.toBeInTheDocument();
        expect(screen.getByRole('textbox', { name: 'Note' })).toHaveValue('');
        expect(screen.getByRole('textbox', { name: /Your name/ })).toHaveValue(' alice ');
    });

    it.each([null, false, 0, '', undefined])('retains note for falsey result %s', async result => {
        /** Failed or absent disposition results must preserve the draft note. */
        const onApplyDisposition = vi.fn().mockResolvedValue(result);
        setup({ onApplyDisposition });
        fill('alice', 'draft');
        await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' })));
        expect(screen.getByRole('textbox', { name: 'Note' })).toHaveValue('draft');
    });

    it('passes undefined for blank notes and honors externally controlled pending state', async () => {
        /** Pending disables actions without disabling editable audit fields. */
        const view = setup({ actionPending: true });
        fill('alice', '   ');
        expect(screen.getByRole('button', { name: 'Acknowledge' })).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' }));
        expect(view.props.onApplyDisposition).not.toHaveBeenCalled();
        expect(screen.getByRole('textbox', { name: 'Note' })).toBeEnabled();
        view.rerender({ actionPending: false });
        await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' })));
        expect(view.props.onApplyDisposition).toHaveBeenCalledWith('acknowledge', 'alice', undefined);
    });

    it('preserves actor, note and validation across close, loading and alert changes', () => {
        /** Modal-local state survives conditionally unmounted detail and shell contents. */
        const view = setup();
        fill(' ', 'draft');
        fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' }));
        fill('alice', 'draft');
        view.rerender({ alertId: null, detail: null });
        view.rerender({ alertId: 8, loading: true, detail: detail({ id: 8 }) });
        view.rerender({ loading: false });
        expect(screen.getByRole('textbox', { name: /Your name/ })).toHaveValue('alice');
        expect(screen.getByRole('textbox', { name: 'Note' })).toHaveValue('draft');
        expect(screen.getByText(/Enter your name so/)).toBeInTheDocument();
    });

    it('retains asynchronous truthy-result semantics across an alert switch', async () => {
        /** A pending callback clears the current note on completion as before extraction. */
        let resolve: (value: unknown) => void = () => undefined;
        const pending = new Promise(complete => { resolve = complete; });
        const view = setup({ onApplyDisposition: vi.fn().mockReturnValue(pending) });
        fill('alice', 'first');
        fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' }));
        expect(screen.getByRole('textbox', { name: 'Note' })).toHaveValue('first');
        view.rerender({ alertId: 8, detail: detail({ id: 8 }) });
        fill('alice', 'second');
        await act(async () => resolve(true));
        expect(screen.getByRole('textbox', { name: 'Note' })).toHaveValue('');
    });

    it('keeps owner attribution and history order, timestamp truncation and optional quoted notes', () => {
        /** The audit trail preserves API order and does not invent missing note text. */
        const view = setup();
        expect(screen.getByText(/No disposition recorded yet/)).toBeInTheDocument();
        expect(screen.getByText('No disposition changes recorded yet.')).toBeInTheDocument();
        view.rerender({ detail: detail({ status: 'resolved', owner: 'alice', disposition_history: [
            { status: 'resolved', actor: 'alice', at: '2026-09-10T12:34:56', note: 'done' },
            { status: 'acknowledged', actor: 'bob', at: '2026-09-09T09:12:00', note: '' },
        ] }) });
        expect(screen.getByText('Currently', { exact: false })).toHaveTextContent('Currently resolved by alice.');
        const rows = screen.getAllByRole('listitem');
        expect(rows[0]).toHaveTextContent('alice 2026-09-10 12:34"done"');
        expect(rows[1]).toHaveTextContent('bob 2026-09-09 09:12');
        expect(rows[1]).not.toHaveTextContent('"');
    });
});
