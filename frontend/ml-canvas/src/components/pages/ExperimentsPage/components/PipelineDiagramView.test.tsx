import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { PipelineDiagramView } from './PipelineDiagramView';
import type { JobInfo } from '../../../../core/api/jobs';

// The view mounts MermaidDiagram, which dynamically imports mermaid. Stub the
// library so jsdom never loads the real ~1MB renderer.
vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({
      svg: '<svg xmlns="http://www.w3.org/2000/svg" data-testid="mermaid-svg"></svg>',
    }),
  },
}));

function makeJob(overrides: Partial<JobInfo> & { job_id: string }): JobInfo {
  return {
    pipeline_id: 'pipe_1',
    node_id: 'n1',
    job_type: 'training',
    status: 'completed',
    start_time: null,
    end_time: null,
    error: null,
    result: null,
    created_at: '2026-08-29T00:00:00Z',
    ...overrides,
  } as JobInfo;
}

const DIAGRAM = 'flowchart TD\n    data["Input Data"] --> model(["logistic_regression"])';

describe('PipelineDiagramView', () => {
  it('shows the empty state when no selected run carries a diagram', () => {
    render(<PipelineDiagramView jobs={[makeJob({ job_id: 'j1', metrics: {} as never })]} />);
    expect(screen.getByText(/none of the selected runs carry a pipeline diagram/i)).toBeInTheDocument();
  });

  it('renders a card per run and mounts the mermaid svg for diagram runs', async () => {
    const jobs = [
      makeJob({ job_id: 'j1', pipeline_id: 'pipe_A', metrics: { pipeline_diagram: DIAGRAM } as never }),
      makeJob({ job_id: 'j2', pipeline_id: 'pipe_B', metrics: {} as never }),
    ];
    render(<PipelineDiagramView jobs={jobs} />);

    expect(screen.getByTestId('pipeline-diagram-view')).toBeInTheDocument();
    // One card per selected run, legacy runs annotated rather than dropped.
    expect(await screen.findAllByTestId('mermaid-svg')).toHaveLength(1);
    expect(screen.getByText(/no diagram recorded for this run/i)).toBeInTheDocument();
  });

  it('reads the diagram from result.metrics when top-level metrics are absent', async () => {
    const jobs = [
      makeJob({
        job_id: 'j1',
        pipeline_id: 'pipe_A',
        result: { metrics: { pipeline_diagram: DIAGRAM } },
      }),
    ];
    render(<PipelineDiagramView jobs={jobs} />);
    expect(await screen.findAllByTestId('mermaid-svg')).toHaveLength(1);
  });

  it('copies the raw mermaid source to the clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });

    render(
      <PipelineDiagramView
        jobs={[makeJob({ job_id: 'j1', metrics: { pipeline_diagram: DIAGRAM } as never })]}
      />
    );

    fireEvent.click(await screen.findByTestId('copy-mermaid-button'));
    expect(writeText).toHaveBeenCalledWith('```mermaid\n' + DIAGRAM + '\n```\n');
    expect(await screen.findByText('Copied!')).toBeInTheDocument();
  });

  it('offers no copy button for legacy runs without a diagram', () => {
    render(<PipelineDiagramView jobs={[makeJob({ job_id: 'j1', metrics: {} as never })]} />);
    expect(screen.queryByTestId('copy-mermaid-button')).not.toBeInTheDocument();
  });
});
