import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { PreviewResponse } from '../api/client';
import { monitoringApi } from '../api/monitoring';
import { useGraphStore } from '../store/useGraphStore';
import { useNotificationsStore } from '../store/useNotificationsStore';
import { OPEN_BELL_EVENT, useExecutionWarnings } from './useExecutionWarnings';

const preview = (overrides: Partial<PreviewResponse> = {}): PreviewResponse => ({
  pipeline_id: 'pipeline', status: 'failed', node_results: {}, preview_data: null,
  recommendations: [], ...overrides,
});

beforeEach(() => {
  useGraphStore.setState({ executionResult: null });
  useNotificationsStore.getState().clear();
  vi.spyOn(monitoringApi, 'logPipelineRun').mockResolvedValue(undefined);
});

afterEach(() => { vi.restoreAllMocks(); });

/** Warnings must reach the persistent bell buffer before the event and backend batch. */
it('notifies before persisting ordered failures and warnings with exact nullish values', async () => {
  const order: string[] = [];
  const onBell = () => {
    order.push('event');
    expect(useNotificationsStore.getState().items.map(item => item.message)).toEqual(['Warning', 'Global']);
  };
  window.addEventListener(OPEN_BELL_EVENT, onBell);
  vi.mocked(monitoringApi.logPipelineRun).mockImplementation(async () => {
    order.push('persist');
  });
  const warnings = [
    { node_id: '', node_type: '', level: 'warning', logger: 'encoder', message: 'Warning' },
    { node_id: null, node_type: null, level: 'info', logger: 'engine', message: 'Global' },
  ];
  const executionResult = preview({ pipeline_id: '', node_warnings: warnings, node_results: {
    second: { status: 'failed', error: 'Second' }, first: { status: 'failed', error: 'First' },
    empty: { status: 'failed', error: '' }, successful: { status: 'completed', error: 'Ignore' },
  } });
  const { unmount } = renderHook(() => useExecutionWarnings());
  try {
    await act(async () => { useGraphStore.setState({ executionResult }); });
    expect(order).toEqual(['event', 'persist']);
    expect(monitoringApi.logPipelineRun).toHaveBeenCalledExactlyOnceWith('', [
      { node_id: 'second', node_type: null, level: 'error', logger: 'engine', message: 'Second' },
      { node_id: 'first', node_type: null, level: 'error', logger: 'engine', message: 'First' },
      ...warnings,
    ]);
  } finally {
    unmount();
    window.removeEventListener(OPEN_BELL_EVENT, onBell);
  }
});

/** Reusing a result must not persist twice, while a distinct run still persists repeated warnings. */
it('deduplicates result identity independently of the notification message buffer', async () => {
  const warning = { node_id: null, node_type: null, level: 'warning', logger: 'engine', message: 'Again' };
  const executionResult = preview({ node_warnings: [warning, warning] });
  const eventSpy = vi.spyOn(window, 'dispatchEvent');
  const { rerender } = renderHook(() => useExecutionWarnings());
  await act(async () => { useGraphStore.setState({ executionResult }); });
  rerender();
  act(() => { useGraphStore.setState({ executionResult: null }); });
  await act(async () => { useGraphStore.setState({ executionResult }); });
  expect(monitoringApi.logPipelineRun).toHaveBeenCalledTimes(1);
  await act(async () => { useGraphStore.setState({ executionResult: { ...executionResult } }); });
  expect(monitoringApi.logPipelineRun).toHaveBeenCalledTimes(2);
  expect(monitoringApi.logPipelineRun).toHaveBeenLastCalledWith('pipeline', [warning, warning]);
  expect(eventSpy).toHaveBeenCalledTimes(2);
  expect(useNotificationsStore.getState().items).toHaveLength(1);
});

/** Backend outages must not remove the local warning or create unhandled rejections. */
it('swallows persistence failures and skips empty results', async () => {
  vi.mocked(monitoringApi.logPipelineRun).mockRejectedValue(new Error('Offline'));
  renderHook(() => useExecutionWarnings());
  await act(async () => { useGraphStore.setState({ executionResult: preview() }); });
  expect(monitoringApi.logPipelineRun).not.toHaveBeenCalled();
  const warning = { node_id: null, node_type: null, level: 'warning', logger: 'engine', message: 'Retained' };
  await act(async () => { useGraphStore.setState({ executionResult: preview({ node_warnings: [warning] }) }); });
  expect(useNotificationsStore.getState().items[0]).toMatchObject(warning);
  expect(monitoringApi.logPipelineRun).toHaveBeenCalledOnce();
});
