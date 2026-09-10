import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import type { ReactNode } from 'react';
import { ClassificationChartsForSplit } from './ClassificationChartsForSplit';

vi.mock('recharts', () => {
  const chart = ({ children, data }: { children?: ReactNode; data?: unknown }) => <div data-testid="series" data-series={JSON.stringify(data)}>{children}</div>;
  const empty = () => null;
  return { ResponsiveContainer: chart, ComposedChart: chart, BarChart: chart, Bar: empty, XAxis: empty, YAxis: empty, CartesianGrid: empty, Tooltip: empty, Legend: empty, Line: empty, ReferenceLine: empty, ReferenceDot: empty, Area: empty };
});

describe('classification chart public series', () => {
  it('keeps the selected label, threshold matrix, ROC, PR and calibration series aligned', () => {
    // Encoded classes must retain their human label when selecting probabilities.
    const handleDownload = vi.fn();
    const { container, rerender } = render(<ClassificationChartsForSplit splitName="test" splitData={{ y_true: ['no', 'yes'], y_pred: ['no', 'no'], y_proba: { classes: [0, 1], labels: ['no', 'yes'], values: [[0.8, 0.2], [0.3, 0.7]] } }} selectedRocClass="yes" threshold={0.5} handleDownload={handleDownload} downloadingChart={null} doneChart={null} />);
    const series = Array.from(container.querySelectorAll('[data-series]'), node => JSON.parse(node.getAttribute('data-series')!));
    expect(series).toContainEqual([{ fpr: 0, tpr: 0, random: 0 }, { fpr: 0, tpr: 1, random: 0 }, { fpr: 1, tpr: 1, random: 1 }]);
    expect(series).toContainEqual([{ recall: 0, precision: 1, score: 1, noSkill: 0.5 }, { recall: 1, precision: 1, score: 0.7, noSkill: 0.5 }, { recall: 1, precision: 0.5, score: 0.2, noSkill: 0.5 }]);
    expect(series).toContainEqual([{ midpoint: 0.25, fracPos: 0, count: 1, perfect: 0.25 }, { midpoint: 0.75, fracPos: 1, count: 1, perfect: 0.75 }]);
    expect(container.querySelector('[title^="True: 1, Pred: 1"]')).toHaveTextContent('1');
    fireEvent.click(screen.getAllByTitle('Download Graph')[0]!);
    expect(handleDownload).toHaveBeenCalledWith('test-confusion-matrix', 'test_confusion_matrix');
    rerender(<ClassificationChartsForSplit splitName="test" splitData={{ y_true: [], y_pred: [] }} selectedRocClass={null} threshold={0} handleDownload={handleDownload} downloadingChart={null} doneChart={null} />);
    expect(container.querySelectorAll('[data-series]')).toHaveLength(0);
  });
});
