import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ColumnProfile } from '../../core/types/edaProfile';
import { getDtypeHexColor } from '../../core/utils/dtypeVisuals';
import { VariableCard } from './VariableCard';

const chart = vi.hoisted(() => ({
  data: [] as { name: string; count: number }[],
  bar: {} as { dataKey?: string; fill?: string; radius?: number[] },
}));
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  BarChart: ({ data, children }: { data: typeof chart.data; children: React.ReactNode }) => {
    chart.data = data;
    return <div data-testid="mini-chart">{children}</div>;
  },
  Bar: (props: typeof chart.bar) => { chart.bar = props; return null; },
}));

const profile: ColumnProfile = { name: 'age', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 };
const histogram = [{ start: 1.24, end: 2, count: 0 }, { start: 2.86, end: 4, count: 7 }];

describe('VariableCard public behavior', () => {
  beforeEach(() => { chart.data = []; chart.bar = {}; });

  it.each([
    ['Numeric', ['1.2', '2.9']],
    ['Text', ['1', '3']],
    ['DateTime', [new Date(1.24).toLocaleDateString(), new Date(2.86).toLocaleDateString()]],
  ] as const)('retains %s histogram labels, order, zero bins and chart styling', (dtype, labels) => {
    // Preview labels and series must retain the original dtype-specific formatting.
    render(<VariableCard profile={{ ...profile, dtype, histogram }} onClick={vi.fn()} />);
    expect(chart.data).toEqual([{ name: labels[0], count: 0 }, { name: labels[1], count: 7 }]);
    expect(chart.bar).toMatchObject({ dataKey: 'count', fill: getDtypeHexColor(dtype), radius: [2, 2, 0, 0] });
  });

  it('uses the first five categorical values even when histogram data is present', () => {
    // Category order and empty labels come from the report, not sorting or histogram fallback.
    const values = ['', 'z', 'a', 'c', 'b', 'sixth'];
    render(<VariableCard profile={{ ...profile, dtype: 'Categorical', histogram, categorical_stats: {
      unique_count: 6, rare_labels_count: 0, top_k: values.map((value, count) => ({ value, count })),
    } }} onClick={vi.fn()} />);
    expect(chart.data).toEqual(values.slice(0, 5).map((name, count) => ({ name, count })));
    expect(screen.getByTestId('mini-chart')).toBeInTheDocument();
  });

  it.each(['Boolean', 'unknown', 'numeric'])('does not invent a histogram for %s', dtype => {
    // Only the existing, case-sensitive supported dtypes produce a mini chart.
    render(<VariableCard profile={{ ...profile, dtype, histogram }} onClick={vi.fn()} />);
    expect(screen.queryByTestId('mini-chart')).not.toBeInTheDocument();
  });

  it.each([undefined, null, []])('omits charts for missing or empty bins: %s', bins => {
    // Optional profiler output must not produce an empty chart container.
    const nextProfile: ColumnProfile = { ...profile };
    if (bins !== undefined) nextProfile.histogram = bins;
    render(<VariableCard profile={nextProfile} onClick={vi.fn()} />);
    expect(screen.queryByTestId('mini-chart')).not.toBeInTheDocument();
  });

  it('renders combined status flags and exact normality precision', () => {
    // Multiple findings coexist; normality does not suppress missingness or identity flags.
    const { rerender } = render(<VariableCard profile={{ ...profile, missing_percentage: 12.345,
      is_unique: true, is_constant: true, normality_test: { is_normal: true, p_value: 0.12345 },
    }} onClick={vi.fn()} />);
    expect(screen.getByText('12.3% null')).toBeInTheDocument();
    expect(screen.getByText('Unique ID')).toBeInTheDocument();
    expect(screen.getByText('Constant')).toBeInTheDocument();
    expect(screen.getByTitle('Normal Distribution (p=0.123)')).toHaveTextContent('Normal Dist.');
    expect(screen.queryByText('Healthy')).not.toBeInTheDocument();
    rerender(<VariableCard profile={{ ...profile, normality_test: { is_normal: false, p_value: 0 } }} onClick={vi.fn()} />);
    expect(screen.getByTitle('Not Normal Distribution (p=0.000)')).toHaveTextContent('Not Normal Dist.');
    expect(screen.queryByText('Healthy')).not.toBeInTheDocument();
  });

  it.each([0, -1, Number.NaN])('retains the exact Healthy condition for missing percentage %s', missing => {
    // Zero alone earns Healthy; negative/non-finite values retain the previous empty status.
    render(<VariableCard profile={{ ...profile, missing_percentage: missing }} onClick={vi.fn()} />);
    expect(screen.queryByText('Healthy') !== null).toBe(missing === 0);
    expect(screen.queryByText(/% null/)).not.toBeInTheDocument();
  });

  it('preserves card click, Enter and Space activation without invoking exclusion', () => {
    // Keyboard and mouse users must reach the same details action.
    const onClick = vi.fn();
    const onToggleExclude = vi.fn();
    render(<VariableCard profile={profile} onClick={onClick} onToggleExclude={onToggleExclude} />);
    const card = screen.getByRole('heading', { name: 'age' }).closest('[role="button"]')!;
    fireEvent.click(card);
    fireEvent.keyDown(card, { key: 'Enter' });
    fireEvent.keyDown(card, { key: ' ' });
    fireEvent.keyDown(card, { key: 'Escape' });
    expect(card).toHaveAttribute('tabindex', '0');
    expect(onClick).toHaveBeenCalledTimes(3);
    expect(onToggleExclude).not.toHaveBeenCalled();
  });

  it('keeps exclusion clicks isolated and uses the current name and excluded state', () => {
    // The toggle must emit its controlled inverse without opening details or retaining old props.
    const onClick = vi.fn();
    const onToggleExclude = vi.fn();
    const { rerender } = render(<VariableCard profile={{ ...profile, histogram }} onClick={onClick} onToggleExclude={onToggleExclude} />);
    const miniChart = screen.getByTestId('mini-chart');
    rerender(<VariableCard profile={{ ...profile, histogram, missing_percentage: 1 }} onClick={onClick} onToggleExclude={onToggleExclude} />);
    expect(screen.getByTestId('mini-chart')).toBe(miniChart);
    fireEvent.click(screen.getByRole('button', { name: 'Exclude from analysis' }));
    expect(onToggleExclude).toHaveBeenLastCalledWith('age', true);
    expect(onClick).not.toHaveBeenCalled();
    rerender(<VariableCard profile={{ ...profile, name: 'height', histogram }} onClick={onClick} onToggleExclude={onToggleExclude} isExcluded />);
    expect(screen.getByText('Excluded from analysis')).toBeInTheDocument();
    expect(screen.queryByText('Numeric')).not.toBeInTheDocument();
    expect(screen.queryByText('Healthy')).not.toBeInTheDocument();
    expect(screen.queryByTestId('mini-chart')).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'height' })).toHaveClass('line-through');
    fireEvent.click(screen.getByRole('button', { name: 'Include in analysis' }));
    expect(onToggleExclude).toHaveBeenLastCalledWith('height', false);
    expect(onClick).not.toHaveBeenCalled();
  });

  it('omits the optional toggle while keeping excluded cards actionable', () => {
    // Exclusion is a display state and must not disable the existing details handler.
    const onClick = vi.fn();
    render(<VariableCard profile={profile} onClick={onClick} isExcluded />);
    expect(screen.getAllByRole('button')).toHaveLength(1);
    fireEvent.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalledOnce();
  });
});
