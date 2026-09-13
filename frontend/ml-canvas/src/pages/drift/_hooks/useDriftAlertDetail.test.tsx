import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { monitoringApi, type DriftAlertDetail } from '../../../core/api/monitoring';
import { useDriftAlertDetail } from './useDriftAlertDetail';

/** Keep each response under test control without replacing the hook's state ownership. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

/** Supply a complete alert with an unmistakable owning ID and job. */
function alert(id: number, status: DriftAlertDetail['status'] = 'new'): DriftAlertDetail {
  return { id, job_id: `job-${id}`, severity: 'warning', status,
    evaluation_status: 'completed', disposition_history: [] };
}

beforeEach(() => {
  vi.spyOn(monitoringApi, 'getDriftAlert').mockImplementation(async id => alert(id));
});
afterEach(() => { vi.restoreAllMocks(); });

it('ignores an old detail response after another alert has loaded', async () => {
  // A late GET must not put one alert's evidence beneath another alert's heading.
  const old = deferred<DriftAlertDetail>();
  vi.mocked(monitoringApi.getDriftAlert).mockReturnValueOnce(old.promise);
  const view = renderHook(({ id }) => useDriftAlertDetail(id), { initialProps: { id: 7 } });
  await act(async () => view.rerender({ id: 8 }));
  expect(view.result.current.detail?.id).toBe(8);
  await act(async () => old.resolve(alert(7)));
  expect(view.result.current.detail?.id).toBe(8);
});

it.each(['resolve', 'reject'] as const)('ignores stale disposition %s without releasing the newer action', async completion => {
  // An earlier alert's action must not replace new details, error text, or pending state.
  const old = deferred<DriftAlertDetail>();
  const current = deferred<DriftAlertDetail>();
  vi.spyOn(monitoringApi, 'updateDriftAlertDisposition').mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
  const view = renderHook(({ id }) => useDriftAlertDetail(id), { initialProps: { id: 7 } });
  await act(async () => undefined);
  let oldResult!: ReturnType<typeof view.result.current.applyDisposition>;
  act(() => { oldResult = view.result.current.applyDisposition('acknowledge', 'alice', 'first'); });
  await act(async () => view.rerender({ id: 8 }));
  expect(view.result.current.actionPending).toBe(false);
  let currentResult!: ReturnType<typeof view.result.current.applyDisposition>;
  act(() => { currentResult = view.result.current.applyDisposition('acknowledge', 'bob', 'second'); });
  await act(async () => {
    if (completion === 'resolve') old.resolve(alert(7, 'acknowledged'));
    else old.reject(new Error('old failure'));
    expect(await oldResult).toBeNull();
  });
  expect(view.result.current.detail?.id).toBe(8);
  expect(view.result.current.error).toBeNull();
  expect(view.result.current.actionPending).toBe(true);
  await act(async () => { current.resolve(alert(8, 'acknowledged')); await currentResult; });
  expect(view.result.current.detail).toEqual(alert(8, 'acknowledged'));
  expect(view.result.current.actionPending).toBe(false);
});

it('keeps closed alert state empty after an in-flight read completes', async () => {
  // Closing an investigation invalidates both its pending read and loading indicator.
  const request = deferred<DriftAlertDetail>();
  vi.mocked(monitoringApi.getDriftAlert).mockReturnValueOnce(request.promise);
  const view = renderHook(({ id }) => useDriftAlertDetail(id), { initialProps: { id: 7 as number | null } });
  act(() => view.rerender({ id: null }));
  await act(async () => request.resolve(alert(7)));
  expect(view.result.current.detail).toBeNull();
  expect(view.result.current.loading).toBe(false);
});

it('keeps the new alert loading when an older detail request fails', async () => {
  // A stale read failure must not replace the current investigation's pending feedback.
  const old = deferred<DriftAlertDetail>();
  const current = deferred<DriftAlertDetail>();
  vi.mocked(monitoringApi.getDriftAlert).mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
  const view = renderHook(({ id }) => useDriftAlertDetail(id), { initialProps: { id: 7 } });
  act(() => view.rerender({ id: 8 }));
  await act(async () => old.reject(new Error('old read failed')));
  expect(view.result.current.loading).toBe(true);
  expect(view.result.current.error).toBeNull();
  await act(async () => current.resolve(alert(8)));
  expect(view.result.current.detail?.id).toBe(8);
  expect(view.result.current.loading).toBe(false);
});

it('keeps the newer manual refresh when same-alert responses arrive in reverse order', async () => {
  // Retrying detail retrieval must not accept an earlier snapshot of the same alert.
  const first = deferred<DriftAlertDetail>();
  const second = deferred<DriftAlertDetail>();
  vi.mocked(monitoringApi.getDriftAlert).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
  const { result } = renderHook(() => useDriftAlertDetail(7));
  let refresh!: Promise<void>;
  act(() => { refresh = result.current.refresh(); });
  await act(async () => { second.resolve(alert(7, 'resolved')); await refresh; });
  await act(async () => first.resolve(alert(7)));
  expect(result.current.detail?.status).toBe('resolved');
});

it('returns no disposition result after unmount', async () => {
  // A disposed investigation must not signal success to code that may own a newer draft.
  const request = deferred<DriftAlertDetail>();
  vi.spyOn(monitoringApi, 'updateDriftAlertDisposition').mockReturnValueOnce(request.promise);
  const view = renderHook(() => useDriftAlertDetail(7));
  await act(async () => undefined);
  let action!: ReturnType<typeof view.result.current.applyDisposition>;
  act(() => { action = view.result.current.applyDisposition('acknowledge', 'alice'); });
  view.unmount();
  request.resolve(alert(7, 'acknowledged'));
  expect(await action).toBeNull();
});

it.each(['resolve', 'reject'] as const)('ignores a pending detail %s after a successful disposition', async completion => {
  // An acknowledged alert must not revert to a pre-action snapshot or a stale read error.
  const read = deferred<DriftAlertDetail>();
  vi.spyOn(monitoringApi, 'updateDriftAlertDisposition').mockResolvedValue(alert(7, 'acknowledged'));
  const { result } = renderHook(() => useDriftAlertDetail(7));
  await act(async () => undefined);
  vi.mocked(monitoringApi.getDriftAlert).mockReturnValueOnce(read.promise);
  let refresh!: Promise<void>;
  act(() => { refresh = result.current.refresh(); });
  await act(async () => { await result.current.applyDisposition('acknowledge', 'alice'); });
  expect(result.current.detail?.status).toBe('acknowledged');
  expect(result.current.loading).toBe(false);
  await act(async () => {
    if (completion === 'resolve') read.resolve(alert(7));
    else read.reject(new Error('pre-action read failure'));
    await refresh;
  });
  expect(result.current.detail?.status).toBe('acknowledged');
  expect(result.current.error).toBeNull();
  expect(result.current.loading).toBe(false);
});
