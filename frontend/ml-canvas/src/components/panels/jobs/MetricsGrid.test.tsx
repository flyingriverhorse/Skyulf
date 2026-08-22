import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { MetricsGrid } from './MetricsGrid';

describe('MetricsGrid', () => {
  const mixedMetrics = {
    test_f1_weighted: 0.91,
    train_f1_weighted: 0.97,
    cv_f1_weighted_mean: 0.9,
    cv_f1_weighted_std: 0.012,
    best_score: 0.9,
  };

  it('shows every metric tile, including CV ones, by default', () => {
    render(<MetricsGrid metrics={mixedMetrics} />);
    expect(screen.getByText('test f1 weighted')).toBeInTheDocument();
    expect(screen.getByText('cv f1 weighted mean')).toBeInTheDocument();
    expect(screen.getByText('best score')).toBeInTheDocument();
  });

  it('hides only the CV tiles when the CV toggle is unchecked', () => {
    render(<MetricsGrid metrics={mixedMetrics} />);
    fireEvent.click(screen.getByRole('checkbox', { name: /cv metrics/i }));
    expect(screen.getByText('test f1 weighted')).toBeInTheDocument();
    expect(screen.getByText('train f1 weighted')).toBeInTheDocument();
    expect(screen.queryByText('cv f1 weighted mean')).not.toBeInTheDocument();
    expect(screen.queryByText('cv f1 weighted std')).not.toBeInTheDocument();
    // best_score is a CV-population metric (the tuning optimum over folds),
    // so it follows the same toggle.
    expect(screen.queryByText('best score')).not.toBeInTheDocument();
  });

  it('shows the CV tiles again when the toggle is re-checked', () => {
    render(<MetricsGrid metrics={mixedMetrics} />);
    const toggle = screen.getByRole('checkbox', { name: /cv metrics/i });
    fireEvent.click(toggle);
    fireEvent.click(toggle);
    expect(screen.getByText('cv f1 weighted mean')).toBeInTheDocument();
  });

  it('offers no toggle when the job reported no CV metrics', () => {
    render(<MetricsGrid metrics={{ test_f1_weighted: 0.91 }} />);
    expect(screen.queryByRole('checkbox', { name: /cv metrics/i })).not.toBeInTheDocument();
    expect(screen.getByText('test f1 weighted')).toBeInTheDocument();
  });

  it('drops keys the caller excludes (non-metric result fields like best_params)', () => {
    render(
      <MetricsGrid
        metrics={{ best_score: 0.9, best_params: 'noise', trials: 50, test_r2: 0.8 }}
        excludeKeys={['best_score', 'best_params', 'trials']}
      />,
    );
    expect(screen.queryByText('best score')).not.toBeInTheDocument();
    expect(screen.queryByText('best params')).not.toBeInTheDocument();
    expect(screen.getByText('test r2')).toBeInTheDocument();
  });
});
