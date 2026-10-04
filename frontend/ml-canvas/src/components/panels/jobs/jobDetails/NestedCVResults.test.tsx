import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { NestedCVResults } from './NestedCVResults';

/** Stored split evidence and independent thresholds must remain visible after reload. */
it('shows group isolation evidence and training-only thresholds', () => {
  render(<NestedCVResults result={{ nested_cv: { status: 'nested_cv', outer_folds: 2, inner_folds: 2,
    scoring_metric: 'accuracy', mean_score: 0.8, std_score: 0.1, total_trials: 3,
    split_policy: { method: 'group_k_fold', group_column: 'customer', gap: 0 },
    threshold_selection: { threshold: 0.4, provenance: 'inner_oof', decision_thresholds: { yes: 0.4 } },
    folds: [{ fold: 1, inner_best_score: 0.9, outer_score: 0.8, best_params: { C: 1 },
      split: { train_groups: 4, test_groups: 2, group_overlap: 0 }, threshold_selection: { threshold: 0.3, decision_thresholds: { yes: 0.3 } } }],
  } }} />);
  expect(screen.getByText(/group_k_fold/)).toBeVisible();
  expect(screen.getByText(/Final training threshold/)).toBeVisible();
  expect(screen.getByText(/"yes":0.3/)).toBeVisible();
  expect(screen.getByText(/"group_overlap": 0/)).toBeInTheDocument();
});
