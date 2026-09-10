import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { JobCard } from './JobCard';
import type { JobInfo } from '../../../core/api/jobs';

/** Keep the real score resolver and badges in the public card boundary. */
const makeJob = (overrides: Partial<JobInfo> = {}): JobInfo => ({
  job_id: 'job-1', pipeline_id: 'pipeline', node_id: 'trainer', job_type: 'training',
  status: 'completed', start_time: null, end_time: null, error: null, result: null,
  created_at: '2026-09-07T12:00:00Z', model_type: 'logistic_regression', ...overrides,
});

describe('JobCard presentation', () => {
  it.each([
    [{ test_accuracy: 0.12345, val_accuracy: 0.4, train_accuracy: 0.8 }, 'test', 'Accuracy: 0.123 (test)'],
    [{ val_accuracy: 0.45678, train_accuracy: 0.8 }, 'val', 'Accuracy: 0.457 (val)'],
    [{ train_accuracy: 0.98765 }, 'train', 'Accuracy: 0.988 (train)'],
    [{ train_f1_weighted: 0.76543, test_accuracy: 0.9 }, 'train', 'F1 Weighted: 0.765 (train)'],
  ] as const)('preserves metric priority and split precision for %s', (metrics, split, label) => {
    // Metric priority precedes split priority; held-out accuracy cannot replace train F1.
    render(<JobCard job={makeJob({ result: { metrics } })} registryItems={[]} onClick={vi.fn()} />);
    expect(screen.getByTitle(`${split} split`)).toHaveTextContent(label);
  });

  it('uses four decimal CV precision and the tuned metric', () => {
    // Tuning scores must retain their CV identity instead of masquerading as held-out scores.
    render(<JobCard job={makeJob({ job_type: 'tuning', search_strategy: 'optuna',
      result: { best_score: 0.123456, scoring_metric: 'accuracy', metrics: { test_accuracy: 0.9 } },
    })} registryItems={[]} onClick={vi.fn()} />);
    expect(screen.getByTitle('cv split')).toHaveTextContent('Accuracy: 0.1235');
    expect(screen.getByTitle('cv split')).not.toHaveTextContent('(cv)');
    expect(screen.getByText('(optuna)')).toBeInTheDocument();
  });

  it.each(['pending', 'queued', 'running', 'failed', 'cancelled', 'succeeded'] as const)(
    'does not display a completed score for %s', status => {
      // Only the exact completed status currently enables the score presentation.
      render(<JobCard job={makeJob({ status, result: { metrics: { test_accuracy: 0.9 } } })}
        registryItems={[]} onClick={vi.fn()} />);
      expect(screen.queryByTitle('test split')).not.toBeInTheDocument();
      expect(screen.getAllByText('-')).toHaveLength(4);
    },
  );

  it('gives an error precedence over completed scores and tuning parameters', () => {
    // A recorded error must remain visible even when stale successful results exist.
    render(<JobCard job={makeJob({ error: 'Training exploded', job_type: 'tuning',
      result: { best_score: 0.9, best_params: {} },
    })} registryItems={[]} onClick={vi.fn()} />);
    expect(screen.getByTitle('Training exploded')).toHaveTextContent('Error');
    expect(screen.queryByTitle('cv split')).not.toBeInTheDocument();
    expect(screen.queryByText('Params found')).not.toBeInTheDocument();
  });

  it.each([
    [{ job_type: 'tuning', result: { best_params: {} } }, true],
    [{ search_strategy: 'grid', result: { best_params: { C: 1 } } }, true],
    [{ result: { best_params: { C: 1 } } }, false],
    [{ job_type: 'tuning', result: { best_params: null } }, false],
    [{ job_type: 'tuning', result: null }, false],
  ] satisfies [Partial<JobInfo>, boolean][])('limits the parameter fallback for %s', (overrides, visible) => {
    // Parameter payloads alone do not turn basic jobs into tuned results.
    render(<JobCard job={makeJob(overrides)} registryItems={[]} onClick={vi.fn()} />);
    expect(Boolean(screen.queryByText('Params found'))).toBe(visible);
  });

  it.each([
    ['voting_classifier', 'Voting · Classification', 'accuracy', 'Accuracy'],
    ['stacking_regressor', 'Stacking · Regression', 'r2', 'R²'],
  ])('resolves ensemble identity and metric task for %s', (model_type, badge, metric, label) => {
    // Ensemble scores must use the underlying task while preserving the strategy badge.
    render(<JobCard job={makeJob({ model_type, engine: 'polars',
      result: { metrics: { [`test_${metric}`]: 0.87654 } },
    })} registryItems={[]} onClick={vi.fn()} />);
    expect(screen.getByText(badge)).toBeInTheDocument();
    expect(screen.getByTitle('Trained on the Polars engine')).toHaveTextContent('Polars');
    expect(screen.getByTitle('test split')).toHaveTextContent(`${label}: 0.877 (test)`);
  });

  it('falls back through dataset ID and missing identity/time fields', () => {
    // Incomplete queued snapshots must remain readable without inventing model or time values.
    const incomplete = makeJob({ dataset_id: 'legacy' });
    delete incomplete.model_type;
    const { rerender } = render(<JobCard job={incomplete}
      registryItems={[]} onClick={vi.fn()} />);
    expect(screen.getByText('Unknown Model')).toBeInTheDocument();
    expect(screen.getByTitle('legacy')).toHaveTextContent('legacy');
    rerender(<JobCard job={makeJob({ dataset_id: 'legacy', dataset_name: 'Named data', engine: 'pandas',
      start_time: '2026-09-07T12:00:00Z', end_time: '2026-09-07T12:01:05Z',
    })} registryItems={[]} onClick={vi.fn()} />);
    expect(screen.getByTitle('Named data')).toHaveTextContent('Named data');
    expect(screen.getByText('1m 5s')).toBeInTheDocument();
    expect(screen.queryByText('Polars')).not.toBeInTheDocument();
  });

  it('activates on click, Enter and Space only', () => {
    // The whole row must remain keyboard accessible after presentation extraction.
    const onClick = vi.fn();
    render(<JobCard job={makeJob()} registryItems={[]} onClick={onClick} />);
    const row = screen.getByRole('button');
    expect(row).toHaveAttribute('tabindex', '0');
    fireEvent.click(row);
    fireEvent.keyDown(row, { key: 'Enter' });
    fireEvent.keyDown(row, { key: ' ' });
    fireEvent.keyDown(row, { key: 'ArrowDown' });
    expect(onClick).toHaveBeenCalledTimes(3);
  });
});
