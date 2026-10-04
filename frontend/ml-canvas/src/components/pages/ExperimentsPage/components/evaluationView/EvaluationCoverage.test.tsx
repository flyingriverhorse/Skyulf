import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { EvaluationCoverage } from './EvaluationCoverage';

describe('EvaluationCoverage', () => {
  it('shows both the evaluated denominator and excluded rows, including empty splits', () => {
    // A filtered metric must not look like an evaluation of every requested row.
    render(<EvaluationCoverage data={{ problem_type: 'regression', splits: {
      test: { y_true: [1, 2], y_pred: [1, 2], coverage: { input_rows: 3, scored_rows: 2, excluded_rows: 1 } },
      validation: { y_true: [], y_pred: [], coverage: { input_rows: 4, scored_rows: 0, excluded_rows: 4 } },
    } }} />);
    expect(screen.getByText(/Test: 2 of 3 rows scored; 1 excluded/)).toBeInTheDocument();
    expect(screen.getByText(/Validation: 0 of 4 rows scored; 4 excluded/)).toBeInTheDocument();
  });

  it('does not invent denominators for older reports', () => {
    // A historical artifact without coverage contains no evidence of the pre-filter population.
    const { container } = render(<EvaluationCoverage data={{ problem_type: 'regression', splits: {
      test: { y_true: [1], y_pred: [1] },
    } }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('reports an unknown merged population without inventing zero excluded rows', () => {
    // Different branch filters cannot be combined into a trustworthy denominator.
    render(<EvaluationCoverage data={{ problem_type: 'regression', splits: {
      test: { y_true: [1], y_pred: [1], coverage: {
        input_rows: null, scored_rows: 1, excluded_rows: null,
        reason: 'Original evaluation population unavailable after merging preprocessing branches.',
      } },
    } }} />);
    expect(screen.getByText(/Test: 1 rows scored; original population and excluded count unavailable/)).toBeInTheDocument();
    expect(screen.queryByText(/0 excluded/)).not.toBeInTheDocument();
  });

  it('keeps partial historical coverage readable without crashing the evaluation tab', () => {
    // Older or partial JSON records may omit counts declared by the current schema.
    render(<EvaluationCoverage data={JSON.parse('{"problem_type":"regression","splits":{"test":{"y_true":[],"y_pred":[],"coverage":{"scored_rows":2}},"validation":{"y_true":[],"y_pred":[],"coverage":{}}}}')} />);
    expect(screen.getByText(/Test: 2 rows scored; original population and excluded count unavailable/)).toBeInTheDocument();
    expect(screen.getByText(/Validation: Scored row count unavailable/)).toBeInTheDocument();
  });
});
