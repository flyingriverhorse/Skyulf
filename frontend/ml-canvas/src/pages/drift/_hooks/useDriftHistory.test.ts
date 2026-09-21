import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { monitoringApi, type DriftHistoryEntry } from '../../../core/api/monitoring';
import { useDriftHistory } from './useDriftHistory';

/** API histories arrive newest first, including checks without per-column measurements. */
function entry(id: number, summary?: DriftHistoryEntry['summary']): DriftHistoryEntry {
    return { id, job_id: 'a', severity: 'none', status: 'new', evaluation_status: 'completed',
        ...(summary === undefined ? {} : { summary }) };
}

afterEach(() => vi.restoreAllMocks());

it('retains missing timestamps without turning absent PSI into measured zero', async () => {
    // Column omission and missing summary must occupy their original chronological slots.
    vi.spyOn(monitoringApi, 'getDriftHistory').mockResolvedValue([
        entry(4, { category: { drifted: true, psi: 0.48 }, numeric: { drifted: false, psi: 0 } }),
        entry(3),
        entry(2, { numeric: { drifted: false, psi: 0.1 }, category: { drifted: false } }),
        entry(1, { category: { drifted: false, psi: 0.2 } }),
    ]);
    const { result } = renderHook(() => useDriftHistory('a'));
    await waitFor(() => expect(result.current.driftHistory).toHaveLength(4));
    expect(result.current.columnSparklines).toEqual({ category: [0.2, null, null, 0.48], numeric: [null, 0.1, null, 0] });
});

it('does not restore another job history when its request finishes late', async () => {
    // A job switch cannot show plausible-looking measurements from a previous model.
    let finish!: (entries: DriftHistoryEntry[]) => void;
    vi.spyOn(monitoringApi, 'getDriftHistory')
        .mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }))
        .mockResolvedValueOnce([entry(20), entry(19)]);
    const { result, rerender } = renderHook(({ job }) => useDriftHistory(job), { initialProps: { job: 'a' } });
    rerender({ job: 'b' });
    await waitFor(() => expect(result.current.driftHistory[0]?.id).toBe(20));
    await act(async () => { finish([entry(1)]); });
    expect(result.current.driftHistory.map(row => row.id)).toEqual([20, 19]);
});
