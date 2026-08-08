import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { usePipelineActions } from './usePipelineActions';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { ConfirmProvider } from '../../../shared';
import { FIT_VIEW_EVENT } from '../../../../core/hooks/useKeyboardShortcuts';
import {
  clearRecentPipelines,
  pushRecentPipeline,
} from '../../../../core/utils/recentPipelines';
import type { PipelineVersionEntry } from '../../../../core/api/pipelineVersions';

vi.mock('../../../../core/api/client', () => ({
  apiClient: { post: vi.fn() },
  savePipeline: vi.fn(),
}));

vi.mock('../../../../core/api/pipelineVersions', () => ({
  pipelineVersionsApi: {
    list: vi.fn().mockResolvedValue([]),
  },
}));

vi.mock('../../../../core/toast', () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}));

// jsdom returns 0x0 rects for the confirm modal's focus-trap probe.
let originalRect: typeof Element.prototype.getBoundingClientRect;
beforeEach(() => {
  originalRect = Element.prototype.getBoundingClientRect;
  Element.prototype.getBoundingClientRect = function () {
    return { width: 100, height: 20, top: 0, left: 0, bottom: 20, right: 100, x: 0, y: 0, toJSON: () => ({}) } as DOMRect;
  };
  useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
  clearRecentPipelines();
});
afterEach(() => {
  Element.prototype.getBoundingClientRect = originalRect;
});

const wrapper: React.FC<{ children: React.ReactNode }> = ({ children }) =>
  React.createElement(ConfirmProvider, null, children);

describe('usePipelineActions recovery flows (CAN-003)', () => {
  it('labels a server version in the overwrite confirmation and focuses on load', async () => {
    const { result } = renderHook(() => usePipelineActions(), { wrapper });
    const entry: PipelineVersionEntry = {
      id: 1,
      datasetId: 'ds-1',
      versionInt: 2,
      name: 'v2 snapshot',
      kind: 'manual',
      pinned: false,
      nodeCount: 1,
      edgeCount: 0,
      createdAt: new Date().toISOString(),
      graph: { nodes: [{ id: 'a', type: 'custom', position: { x: 0, y: 0 }, data: {} }], edges: [] },
    };

    const fitViewSpy = vi.fn();
    window.addEventListener(FIT_VIEW_EVENT, fitViewSpy);

    let loadPromise!: Promise<void>;
    act(() => {
      loadPromise = result.current.handleLoadVersion(entry);
    });
    // Confirm dialog is now open; find and click "Load".
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeTruthy());
    const dialog = document.querySelector('[role="dialog"]') as HTMLElement;
    expect(dialog.textContent).toMatch(/server version/i);
    const confirmButton = Array.from(dialog.querySelectorAll('button')).find(
      (b) => b.textContent === 'Load',
    )!;
    act(() => confirmButton.click());
    await act(async () => loadPromise);

    expect(useGraphStore.getState().nodes).toHaveLength(1);
    expect(fitViewSpy).toHaveBeenCalledTimes(1);
    window.removeEventListener(FIT_VIEW_EVENT, fitViewSpy);
  });

  it('labels a local-recent snapshot in its overwrite confirmation and cancel leaves the canvas untouched', async () => {
    const entries = pushRecentPipeline({ name: 'My snapshot', nodes: [{ id: 'x', type: 'custom', position: { x: 0, y: 0 }, data: {} }], edges: [] });
    const { result } = renderHook(() => usePipelineActions(), { wrapper });

    let restorePromise!: Promise<void>;
    act(() => {
      restorePromise = result.current.handleRestoreRecent(entries[0]!);
    });
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeTruthy());
    const dialog = document.querySelector('[role="dialog"]') as HTMLElement;
    expect(dialog.textContent).toMatch(/local recent/i);
    const cancelButton = Array.from(dialog.querySelectorAll('button')).find(
      (b) => b.textContent === 'Cancel',
    )!;
    act(() => cancelButton.click());
    await act(async () => restorePromise);

    expect(useGraphStore.getState().nodes).toEqual([]);
  });
});
