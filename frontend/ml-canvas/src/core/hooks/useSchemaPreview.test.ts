import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../api/client';
import type { SchemaPreviewResponse } from '../api/schemaPreview';
import { useGraphStore } from '../store/useGraphStore';
import { useSchemaPreview } from './useSchemaPreview';

/** Control the external HTTP response independently of effect cleanup. */
function pendingResponse() {
  let resolve!: (value: { data: SchemaPreviewResponse }) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<{ data: SchemaPreviewResponse }>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Use real graph conversion and store writes around the mocked HTTP boundary. */
function setDataset(id: string) {
  useGraphStore.setState({
    nodes: [{ id, position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: id } }],
    edges: [],
  });
}

const response: SchemaPreviewResponse = {
  pipeline_id: 'preview',
  predicted_schemas: { data: { columns: ['age'], dtypes: { age: 'int64' } } },
  broken_references: [{ node_id: 'model', field: 'target', column: 'missing', upstream_node_id: 'data' }],
};

beforeEach(() => {
  vi.useFakeTimers();
  setDataset('data');
  useGraphStore.setState({ predictedSchemas: {}, brokenSchemaRefs: {} });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('useSchemaPreview request lifetime', () => {
  it('debounces the graph and stores the current schema and grouped references', async () => {
    // A current successful response must keep the canvas badges and reference warnings working.
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: response });
    renderHook(useSchemaPreview);
    await act(async () => { await vi.advanceTimersByTimeAsync(399); });
    expect(post).not.toHaveBeenCalled();
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(useGraphStore.getState().predictedSchemas).toEqual(response.predicted_schemas);
    expect(useGraphStore.getState().brokenSchemaRefs).toEqual({
      model: [{ field: 'target', column: 'missing', upstream_node_id: 'data' }],
    });
  });

  it.each(['unmount', 'empty graph', 'replacement graph'] as const)('aborts an in-flight request on %s', async change => {
    // Cleanup must reach the HTTP transport before a replacement debounce starts.
    const pending = pendingResponse();
    const post = vi.spyOn(apiClient, 'post').mockReturnValue(pending.promise);
    const view = renderHook(useSchemaPreview);
    await act(async () => { await vi.advanceTimersByTimeAsync(400); });
    const signal = post.mock.calls[0]?.[2]?.signal;
    act(() => {
      if (change === 'unmount') view.unmount();
      else if (change === 'empty graph') useGraphStore.setState({ nodes: [], edges: [] });
      else setDataset('replacement');
    });
    expect(signal?.aborted).toBe(true);
  });

  it.each(['unmount', 'empty graph', 'replacement graph'] as const)('ignores a late successful response after %s', async change => {
    // Even a transport that finishes after cancellation must not repopulate stale graph state.
    const pending = pendingResponse();
    vi.spyOn(apiClient, 'post').mockReturnValue(pending.promise);
    const view = renderHook(useSchemaPreview);
    await act(async () => { await vi.advanceTimersByTimeAsync(400); });
    act(() => {
      if (change === 'unmount') view.unmount();
      else if (change === 'empty graph') useGraphStore.setState({ nodes: [], edges: [] });
      else setDataset('replacement');
    });
    await act(async () => { pending.resolve({ data: response }); });
    expect(useGraphStore.getState().predictedSchemas).toEqual({});
    expect(useGraphStore.getState().brokenSchemaRefs).toEqual({});
  });

  it('silences a canceled request failure', async () => {
    // Leaving the canvas is expected cleanup and must not be logged as an API failure.
    const pending = pendingResponse();
    vi.spyOn(apiClient, 'post').mockReturnValue(pending.promise);
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => {});
    const { unmount } = renderHook(useSchemaPreview);
    await act(async () => { await vi.advanceTimersByTimeAsync(400); });
    unmount();
    await act(async () => { pending.reject(new Error('canceled')); });
    expect(debug).not.toHaveBeenCalled();
  });

  it('clears the debounce on unmount before issuing a request', async () => {
    // Leaving before the debounce expires must not start background schema work.
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: response });
    const { unmount } = renderHook(useSchemaPreview);
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(400); });
    expect(post).not.toHaveBeenCalled();
  });
});
