import React, { isValidElement, cloneElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import { FeatureImportanceView, type FeatureImportanceCoverageInput, type FeatureImportanceEntry } from './FeatureImportanceView';

// ResponsiveContainer relies on a real layout engine (via ResizeObserver) to
// size its children, which jsdom doesn't provide — swap it for a passthrough
// that hands the chart explicit numeric dimensions so it (and its
// <defs>/<pattern> hatch marker) still renders in tests.
vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactElement }) => (
      <div style={{ width: 800, height: 500 }}>
        {isValidElement(children) ? cloneElement(children, { width: 800, height: 500 } as never) : children}
      </div>
    ),
  };
});

afterEach(() => {
  document.documentElement.classList.remove('dark');
  Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 1024 });
});

const baseCoverage = (overrides: Partial<FeatureImportanceCoverageInput>): FeatureImportanceCoverageInput => ({
  jobId: 'job-1',
  label: 'random_forest (a1b2c3d4)',
  task: 'classification',
  status: 'completed',
  hasArtifact: true,
  ...overrides,
});

const noop = vi.fn();

describe('FeatureImportanceView', () => {
  it('renders the availability list distinguishing supported, unsupported, not-yet-computed, and failed runs', () => {
    const jobs: FeatureImportanceEntry[] = [
      { jobId: 'a', pipeline_id: 'preview_aaaaaaaa', modelType: 'random_forest', importances: { feature_a: 1 } },
    ];
    const coverageInputs: FeatureImportanceCoverageInput[] = [
      baseCoverage({ jobId: 'a', label: 'random_forest (aaaaaaaa)', hasArtifact: true }),
      baseCoverage({ jobId: 'b', label: 'kmeans (bbbbbbbb)', task: 'segmentation', hasArtifact: false }),
      baseCoverage({ jobId: 'c', label: 'gbm (cccccccc)', status: 'running', hasArtifact: false }),
      baseCoverage({ jobId: 'd', label: 'svm (dddddddd)', status: 'failed', hasArtifact: false }),
    ];

    render(
      <FeatureImportanceView
        featureImportancesByJob={jobs}
        coverageInputs={coverageInputs}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
      />,
    );

    expect(screen.getByText('Available')).toBeInTheDocument();
    expect(screen.getByText('Unsupported')).toBeInTheDocument();
    expect(screen.getByText('Not computed')).toBeInTheDocument();
    expect(screen.getByText('Failed')).toBeInTheDocument();
  });

  it('states the normalization scale explicitly in the chart caption', () => {
    const jobs: FeatureImportanceEntry[] = [
      { jobId: 'a', pipeline_id: 'preview_aaaaaaaa', modelType: 'random_forest', importances: { feature_a: 1, feature_b: 0.5 } },
    ];
    render(
      <FeatureImportanceView
        featureImportancesByJob={jobs}
        coverageInputs={[baseCoverage({ jobId: 'a' })]}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
      />,
    );

    expect(screen.getByText(/min–max normalized per run/i)).toBeInTheDocument();
    expect(screen.getByText(/1\.0 is that run's single largest reported feature/i)).toBeInTheDocument();
  });

  it('provides a data-table alternative with normalized and raw columns per run', () => {
    const jobs: FeatureImportanceEntry[] = [
      { jobId: 'a', pipeline_id: 'preview_aaaaaaaa', modelType: 'random_forest', importances: { feature_a: 4, feature_b: 2 } },
    ];
    render(
      <FeatureImportanceView
        featureImportancesByJob={jobs}
        coverageInputs={[baseCoverage({ jobId: 'a' })]}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /view data table/i }));
    expect(screen.getByText(/normalized \(run max = 1\.0\)/i)).toBeInTheDocument();
    expect(screen.getAllByText(/raw value/i).length).toBeGreaterThan(0);
  });

  it('distinguishes a feature a run does not report from a genuine zero via a non-color hatch cue', () => {
    const jobs: FeatureImportanceEntry[] = [
      { jobId: 'a', pipeline_id: 'preview_aaaaaaaa', modelType: 'random_forest', importances: { feature_a: 1, feature_b: 0 } },
      { jobId: 'b', pipeline_id: 'preview_bbbbbbbb', modelType: 'gbm', importances: { feature_a: 1 } },
    ];
    const { container } = render(
      <FeatureImportanceView
        featureImportancesByJob={jobs}
        coverageInputs={[baseCoverage({ jobId: 'a' }), baseCoverage({ jobId: 'b', label: 'gbm (bbbbbbbb)' })]}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
      />,
    );

    // The hatch <pattern> def used to mark "not reported" bars is present.
    expect(container.querySelector('pattern#feature-importance-not-reported-hatch')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: /view data table/i }));
    expect(screen.getAllByText('not reported').length).toBeGreaterThan(0);
  });

  it('renders correctly in dark mode', () => {
    document.documentElement.classList.add('dark');
    const jobs: FeatureImportanceEntry[] = [
      { jobId: 'a', pipeline_id: 'preview_aaaaaaaa', modelType: 'random_forest', importances: { feature_a: 1 } },
    ];
    render(
      <FeatureImportanceView
        featureImportancesByJob={jobs}
        coverageInputs={[baseCoverage({ jobId: 'a' })]}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
      />,
    );
    expect(screen.getByText('Feature Importance Comparison')).toBeInTheDocument();
  });

  it('renders correctly at a 390px viewport', () => {
    Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 390 });
    window.dispatchEvent(new Event('resize'));
    const jobs: FeatureImportanceEntry[] = [
      { jobId: 'a', pipeline_id: 'preview_aaaaaaaa', modelType: 'random_forest', importances: { feature_a: 1 } },
    ];
    render(
      <FeatureImportanceView
        featureImportancesByJob={jobs}
        coverageInputs={[baseCoverage({ jobId: 'a' })]}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
      />,
    );
    expect(screen.getByText('Feature Importance Comparison')).toBeInTheDocument();
  });

  it('renders only the availability list, and no chart, when no run has data', () => {
    render(
      <FeatureImportanceView
        featureImportancesByJob={[]}
        coverageInputs={[baseCoverage({ jobId: 'a', task: 'segmentation', hasArtifact: false })]}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
      />,
    );
    expect(screen.getByText('Unsupported')).toBeInTheDocument();
    expect(screen.queryByText('Feature Importance Comparison')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /view data table/i })).not.toBeInTheDocument();
  });
});
