import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { jobsApi, type JobInfo } from '../../../core/api/jobs';
import { PipelineDiffView } from './PipelineDiffView';
import * as layout from './pipelineDiffLayout';
import type { JobLite } from './pipelineDiffLayout';

const { flowProps } = vi.hoisted(() => ({ flowProps: vi.fn() }));

vi.mock('../../../core/api/jobs', () => ({ jobsApi: { getJob: vi.fn() } }));
vi.mock('@xyflow/react', async (importOriginal) => ({
  ...await importOriginal<typeof import('@xyflow/react')>(),
  Background: () => null,
  Controls: () => null,
  ReactFlowProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  ReactFlow: (props: Record<string, unknown>) => {
    flowProps(props);
    return <div data-testid="graph" />;
  },
}));

const first: JobLite = {
  job_id: 'job-first', pipeline_id: 'first-pipeline', dataset_name: 'Alpha',
  model_type: 'forest', created_at: '2026-01-01T08:30:00Z',
};
const second: JobLite = { job_id: 'job-second', pipeline_id: 'second-pipeline' };
const third: JobLite = { job_id: 'job-third', pipeline_id: 'third-pipeline', created_at: 'invalid-time' };
const selected = [first, second];
const node = (id: string, label: string, method = 'mean') => ({
  id, data: { label, method }, position: { x: 42, y: 77 },
});
const graph = (method = 'mean') => ({ nodes: [node('prep', 'Preparation', method)], edges: [] });
const info = (savedGraph: unknown): JobInfo => ({ graph: savedGraph } as JobInfo);
const deferred = () => {
  let resolve!: (value: JobInfo) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<JobInfo>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
};
const resolvePair = (left: unknown = graph(), right: unknown = graph('median')) => {
  vi.mocked(jobsApi.getJob).mockResolvedValueOnce(info(left)).mockResolvedValueOnce(info(right));
};
const settle = async (callback: () => void) => {
  await act(async () => { callback(); });
};
const baselineCard = () => screen.getByText('Baseline').closest('.overflow-hidden') as HTMLElement;

describe('PipelineDiffView characterization', () => {
  beforeEach(() => { vi.restoreAllMocks(); vi.mocked(jobsApi.getJob).mockReset(); flowProps.mockClear(); });

  it.each([{ jobs: [] }, { jobs: [first] }, { jobs: [first, second, third] }])('requires exactly two selected jobs ($jobs.length)', ({ jobs }) => {
    // Invalid selections must not start requests or show graph viewers.
    render(<PipelineDiffView jobs={jobs} />);
    expect(screen.getByText('Pick exactly two runs')).toBeInTheDocument();
    expect(screen.getByText(`(${jobs.length} selected)`, { exact: false })).toBeInTheDocument();
    expect(jobsApi.getJob).not.toHaveBeenCalled();
    expect(screen.queryByTestId('graph')).not.toBeInTheDocument();
  });

  it('waits for both requests before presenting either snapshot', async () => {
    // Partial completion must keep the shared loading view visible.
    const left = deferred(); const right = deferred();
    vi.mocked(jobsApi.getJob).mockReturnValueOnce(left.promise).mockReturnValueOnce(right.promise);
    render(<PipelineDiffView jobs={selected} />);
    expect(vi.mocked(jobsApi.getJob).mock.calls).toEqual([[first.job_id], [second.job_id]]);
    await settle(() => left.resolve(info(graph())));
    expect(screen.getByText('Loading pipeline graphs…')).toBeInTheDocument();
    expect(screen.queryByTestId('graph')).not.toBeInTheDocument();
    await settle(() => right.resolve(info(graph())));
    expect(screen.getAllByTestId('graph')).toHaveLength(2);
  });

  it.each([[null, null, 2], [null, graph(), 1], [graph(), null, 1]])(
    'names missing snapshots without rendering a partial graph', async (left, right, count) => {
      // One missing side blocks both viewers and keeps the affected run identifiable.
      resolvePair(left, right);
      render(<PipelineDiffView jobs={selected} />);
      expect(await screen.findByText('Pipeline Diff needs two saved snapshots')).toBeInTheDocument();
      expect(screen.getAllByText(/has no saved pipeline snapshot/)).toHaveLength(count as number);
      if (!left) expect(screen.getByText(/Baseline run .*Alpha · forest/)).toBeInTheDocument();
      if (!right) expect(screen.getByText(/Candidate run .*Unknown dataset · Unknown model · Unknown time/)).toBeInTheDocument();
      expect(screen.queryByTestId('graph')).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Swap' })).not.toBeInTheDocument();
    },
  );

  it.each([0, 1])('reports an Error rejection on side %i', async (side) => {
    // API failures remain per-side snapshot issues rather than global errors.
    vi.mocked(jobsApi.getJob).mockImplementation((id) => id === selected[side]!.job_id
      ? Promise.reject(new Error('snapshot unavailable')) : Promise.resolve(info(graph())));
    render(<PipelineDiffView jobs={selected} />);
    expect(await screen.findByText(/could not be loaded: snapshot unavailable/)).toHaveTextContent(side === 0 ? 'Baseline run' : 'Candidate run');
    expect(screen.queryByTestId('graph')).not.toBeInTheDocument();
  });

  it('retains fallback rejection text alongside a missing snapshot', async () => {
    // Non-Error rejections must not leak arbitrary response values into the message.
    vi.mocked(jobsApi.getJob).mockRejectedValueOnce('raw failure').mockResolvedValueOnce(info(null));
    render(<PipelineDiffView jobs={selected} />);
    expect(await screen.findByText(/Baseline run .*could not be loaded: Failed to load job graph/)).toBeInTheDocument();
    expect(screen.getByText(/Candidate run .*has no saved pipeline snapshot/)).toBeInTheDocument();
  });

  it('keeps empty saved graphs ready and retains metadata fallbacks', async () => {
    // Truthy empty snapshots are valid comparisons, including unknown and invalid timestamps.
    resolvePair({}, { nodes: [], edges: [] });
    render(<PipelineDiffView jobs={[second, third]} />);
    expect(await screen.findByText(/No structural or config differences detected/)).toBeInTheDocument();
    expect(screen.getAllByTestId('graph')).toHaveLength(2);
    expect(screen.getByText('Unknown dataset · Unknown model · Unknown time')).toBeInTheDocument();
    expect(screen.getByText('Unknown dataset · Unknown model · invalid-time')).toBeInTheDocument();
    expect(screen.queryByText('Changes')).not.toBeInTheDocument();
  });

  it('preserves directional changes, read-only viewers and shared layout when swapping', async () => {
    // Swap must reverse the actual diff without refetching or changing navigation behavior.
    resolvePair();
    render(<PipelineDiffView jobs={selected} />);
    expect(await screen.findByText('method: "mean" → "median"')).toBeInTheDocument();
    expect(within(baselineCard()).getByText(/Alpha · forest/)).toHaveTextContent(new Date(first.created_at!).toLocaleString());
    const [left, right] = flowProps.mock.calls.slice(-2).map(([props]) => props);
    expect(left.nodes[0].position).toEqual(right.nodes[0].position);
    expect(left.nodes[0]).toMatchObject({ type: 'diff', draggable: false, selectable: false, data: { diffStatus: 'modified' } });
    expect(left).toMatchObject({ fitView: true, fitViewOptions: { padding: 0.2, includeHiddenNodes: false },
      minZoom: 0.2, maxZoom: 1.5, nodesDraggable: false, nodesConnectable: false,
      elementsSelectable: false, panOnScroll: false, zoomOnScroll: false, zoomOnPinch: false,
      zoomOnDoubleClick: false, preventScrolling: false, proOptions: { hideAttribution: true } });
    fireEvent.click(screen.getByRole('button', { name: 'Swap' }));
    expect(screen.getByText('method: "median" → "mean"')).toBeInTheDocument();
    expect(within(baselineCard()).getByText(/Unknown dataset/)).toBeInTheDocument();
    expect(jobsApi.getJob).toHaveBeenCalledTimes(2);
  });

  it('refetches and resets Swap on a new jobs array even when ids match', async () => {
    // The selected-jobs array identity is the existing request and swap-reset dependency.
    resolvePair(); resolvePair();
    const view = render(<PipelineDiffView jobs={selected} />);
    await screen.findByText('method: "mean" → "median"');
    fireEvent.click(screen.getByRole('button', { name: 'Swap' }));
    view.rerender(<PipelineDiffView jobs={selected} />);
    expect(screen.getByText('method: "median" → "mean"')).toBeInTheDocument();
    expect(jobsApi.getJob).toHaveBeenCalledTimes(2);
    view.rerender(<PipelineDiffView jobs={[first, second]} />);
    expect(await screen.findByText('method: "mean" → "median"')).toBeInTheDocument();
    expect(jobsApi.getJob).toHaveBeenCalledTimes(4);
  });

  it.each(['resolve', 'reject'] as const)('ignores stale %s after selection replacement', async (completion) => {
    // Cancelled requests must not overwrite a newer pair or end its loading state.
    const oldLeft = deferred(); const oldRight = deferred(); const currentLeft = deferred(); const currentRight = deferred();
    vi.mocked(jobsApi.getJob).mockReturnValueOnce(oldLeft.promise).mockReturnValueOnce(oldRight.promise)
      .mockReturnValueOnce(currentLeft.promise).mockReturnValueOnce(currentRight.promise);
    const view = render(<PipelineDiffView jobs={selected} />);
    view.rerender(<PipelineDiffView jobs={[second, third]} />);
    await settle(() => {
      if (completion === 'reject') oldLeft.reject(new Error('old failure'));
      else oldLeft.resolve(info(null));
      oldRight.resolve(info(null));
    });
    expect(screen.getByText('Loading pipeline graphs…')).toBeInTheDocument();
    await settle(() => { currentLeft.resolve(info(graph())); currentRight.resolve(info(graph('median'))); });
    expect(screen.getByText('method: "mean" → "median"')).toBeInTheDocument();
    expect(screen.queryByText(/old failure/)).not.toBeInTheDocument();
  });

  it('ignores completion after unmount before normalizing saved graphs', async () => {
    // Cleanup must stop asynchronous processing, not merely suppress a React warning.
    const left = deferred(); const right = deferred();
    const readGraph = vi.spyOn(layout, 'readSideFromGraph');
    vi.mocked(jobsApi.getJob).mockReturnValueOnce(left.promise).mockReturnValueOnce(right.promise);
    const view = render(<PipelineDiffView jobs={selected} />);
    view.unmount();
    await settle(() => { left.resolve(info(graph())); right.resolve(info(graph())); });
    expect(readGraph).not.toHaveBeenCalled();
    expect(flowProps).not.toHaveBeenCalled();
  });

  it('clears pending state when the selection stops containing two jobs', async () => {
    // Reducing the selection must prevent a pending response from restoring the diff.
    const pending = deferred();
    vi.mocked(jobsApi.getJob).mockReturnValue(pending.promise);
    const view = render(<PipelineDiffView jobs={selected} />);
    view.rerender(<PipelineDiffView jobs={[first]} />);
    await settle(() => pending.resolve(info(graph())));
    expect(screen.getByText('Pick exactly two runs')).toBeInTheDocument();
    expect(screen.queryByText('Loading pipeline graphs…')).not.toBeInTheDocument();
  });

  it('renders summary counts and each renamed modified node only once', async () => {
    // Alias entries must collapse while additions, removals and edge direction stay visible.
    resolvePair({ nodes: [node('old-prep', 'Preparation'), node('gone', 'Removed'), node('same', 'Stable')],
      edges: [{ id: 'old-edge', source: 'gone', target: 'old-prep' }] },
    { nodes: [node('new-prep', 'Preparation', 'median'), node('new', 'Added'), node('same', 'Stable')],
      edges: [{ id: 'new-edge', source: 'new-prep', target: 'new' }] });
    render(<PipelineDiffView jobs={selected} />);
    expect(await screen.findByText('1 added')).toBeInTheDocument();
    expect(screen.getByText('1 removed')).toBeInTheDocument();
    expect(screen.getByText('1 modified')).toBeInTheDocument();
    expect(screen.getByText(/1 unchanged \(1 renamed across runs\).*edges 1\+ \/ 1−/)).toBeInTheDocument();
    expect(screen.getAllByText('method: "mean" → "median"')).toHaveLength(1);
    expect(screen.getAllByText('Preparation')).toHaveLength(1);
    expect(screen.getByText('Changes').nextElementSibling?.children).toHaveLength(3);
  });
});
