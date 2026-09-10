import { render, fireEvent, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import type { ReactNode } from 'react';
import { RegressionChartsForSplit } from './RegressionChartsForSplit';

vi.mock('recharts', () => {
  const chart = ({ children, data }: { children?: ReactNode; data?: unknown }) => <div data-series={JSON.stringify(data)}>{children}</div>;
  const empty = () => null;
  return { ResponsiveContainer: chart, ComposedChart: chart, ScatterChart: chart, Scatter: chart, BarChart: chart, Bar: empty, XAxis: empty, YAxis: empty, CartesianGrid: empty, Tooltip: empty, Legend: empty, Line: chart, ReferenceLine: empty };
});

describe('regression chart public series', () => {
  it('preserves residual direction, fit ordering, error percentiles and download identity', () => {
    // Signed errors and input order matter for interpreting fit and lag charts.
    const handleDownload = vi.fn();
    const { container, rerender } = render(<RegressionChartsForSplit splitName="validation" splitData={{ y_true: [3, 0, 2, 1], y_pred: [2, 0, 4, 1] }} handleDownload={handleDownload} downloadingChart={null} doneChart={null} />);
    const series = Array.from(container.querySelectorAll('[data-series]'), node => JSON.parse(node.getAttribute('data-series')!));
    expect(series).toContainEqual([{ x: 3, y: 2 }, { x: 0, y: 0 }, { x: 2, y: 4 }, { x: 1, y: 1 }]);
    expect(series).toContainEqual([{ x: 3, y: 2, residual: 1 }, { x: 0, y: 0, residual: 0 }, { x: 2, y: 4, residual: -2 }, { x: 1, y: 1, residual: 0 }]);
    expect(series).toContainEqual([{ r0: 1, r1: 0 }, { r0: 0, r1: -2 }, { r0: -2, r1: 0 }]);
    expect(screen.getByText('Absolute error percentiles:')).toBeInTheDocument();
    fireEvent.click(screen.getAllByTitle('Download Graph')[0]!);
    expect(handleDownload).toHaveBeenCalledWith('validation-actual-pred', 'validation_actual_vs_predicted');
    rerender(<RegressionChartsForSplit splitName="validation" splitData={{ y_true: [], y_pred: [] }} handleDownload={handleDownload} downloadingChart={null} doneChart={null} />);
    expect(screen.queryByText('Q-Q Plot (Residuals)')).not.toBeInTheDocument();
  });
});
