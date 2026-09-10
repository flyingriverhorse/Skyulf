import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { toPng } from 'html-to-image';
import { VariableRow } from './VariableRow';
import { toast } from '../../core/toast';
import type { ColumnProfile } from '../../core/types/edaProfile';
import type { DistributionDatum } from './DistributionChart';

const chart = vi.hoisted(() => ({ click: undefined as undefined | ((data: DistributionDatum) => void) }));
vi.mock('html-to-image', () => ({ toPng: vi.fn() }));
vi.mock('../../core/toast', () => ({ toast: { error: vi.fn() } }));
vi.mock('./DistributionChart', () => ({ DistributionChart: ({ onBarClick }: { onBarClick: typeof chart.click }) => {
  chart.click = onBarClick;
  return <div data-testid="distribution" />;
} }));
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  BarChart: ({ data, children }: { data: unknown; children: React.ReactNode }) => <div data-testid="mini" data-values={JSON.stringify(data)}>{children}</div>,
  Bar: () => null,
}));

const profile: ColumnProfile = { name: 'age', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 };
const baseProps = { profile, isExpanded: true, onToggleExpand: vi.fn(), onToggleExclude: vi.fn(), isExcluded: false, handleAddFilter: vi.fn() };
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); document.documentElement.classList.remove('dark'); });

describe('VariableRow contracts', () => {
  it('preserves zero and absent statistics, variance fallback and shape boundaries', () => {
    render(<VariableRow {...baseProps} profile={{ ...profile, numeric_stats: { mean: 0, std: 2, q25: null, skewness: 0.5, kurtosis: -0.5, zeros_count: 0, negatives_count: 0 }, vif: 5 }} />);
    expect(screen.getByText('Mean').nextSibling).toHaveTextContent('0.0000');
    expect(screen.getByText('Variance').nextSibling).toHaveTextContent('4.0000');
    expect(screen.getByText('25% (Q1)').nextSibling).toHaveTextContent('');
    expect(screen.getByText(/Moderately Skewed/)).toBeInTheDocument();
    expect(screen.getByText(/Platykurtic/)).toBeInTheDocument();
    expect(screen.getByText('Zeros').nextSibling).toHaveTextContent('0');
    expect(screen.getByText('5.00')).toHaveClass('text-amber-500');
  });

  it('keeps empty histogram precedence over categories and collapsed numeric summary precedence', () => {
    const { rerender } = render(<VariableRow {...baseProps} isExpanded={false} profile={{ ...profile, histogram: [], numeric_stats: { mean: 0 }, categorical_stats: { unique_count: 2, rare_labels_count: 0, top_k: [{ value: 'x', count: 2 }] } }} />);
    expect(screen.queryByTestId('mini')).not.toBeInTheDocument();
    expect(screen.getByText('0.00')).toBeInTheDocument();
    expect(screen.queryByText('Unique:')).not.toBeInTheDocument();
    rerender(<VariableRow {...baseProps} isExpanded={false} profile={{ ...profile, histogram: [{ start: 0, end: 1, count: 2 }] }} />);
    expect(screen.getByTestId('mini')).toHaveAttribute('data-values', '[{"name":"0","count":2}]');
  });

  it.each(['Categorical', 'Boolean', 'Numeric', 'Text', 'constructor'])('keeps %s bar filter payloads', (dtype) => {
    const handleAddFilter = vi.fn();
    render(<VariableRow {...baseProps} profile={{ ...profile, dtype }} handleAddFilter={handleAddFilter} />);
    const datum = { name: 'fallback', fullName: 'full', count: 1, value: 0, rawBin: { start: 0, end: 1, count: 1 } };
    act(() => chart.click?.(datum));
    if (dtype === 'Categorical' || dtype === 'Boolean') expect(handleAddFilter).toHaveBeenCalledWith('age', 0, '==');
    else if (dtype === 'Numeric') expect(handleAddFilter).toHaveBeenCalledWith('age', 0, '>=');
    else expect(handleAddFilter).not.toHaveBeenCalled();
  });

  it('keeps chart DOM identity during rerenders and prevents exclusion from expanding', () => {
    const onToggleExpand = vi.fn();
    const onToggleExclude = vi.fn();
    const { rerender } = render(<VariableRow {...baseProps} onToggleExpand={onToggleExpand} onToggleExclude={onToggleExclude} />);
    const distribution = screen.getByTestId('distribution');
    fireEvent.click(screen.getByRole('button', { name: 'Exclude from analysis' }));
    expect(onToggleExclude).toHaveBeenCalledWith('age', true);
    expect(onToggleExpand).not.toHaveBeenCalled();
    rerender(<VariableRow {...baseProps} isExcluded />);
    expect(screen.getByTestId('distribution')).toBe(distribution);
  });

  it('retains categorical empty-mode fallback, title, zero rare count and top-five limit', () => {
    render(<VariableRow {...baseProps} profile={{ ...profile, dtype: 'Categorical', categorical_stats: {
      unique_count: 6, rare_labels_count: 0,
      top_k: ['', 'second', 'third', 'fourth', 'fifth', 'sixth'].map(value => ({ value, count: 0 })),
    } }} />);
    expect(screen.getByText('Mode').nextSibling).toHaveTextContent('-');
    expect(screen.getByText('Mode').nextSibling).toHaveAttribute('title', '');
    expect(screen.getByText('Rare Labels').nextSibling).toHaveTextContent('0');
    expect(screen.getByText('fifth')).toBeInTheDocument();
    expect(screen.queryByText('sixth')).not.toBeInTheDocument();
  });

  it('keeps text sentiment percentages, nullish word fallback and ten-word limit', () => {
    render(<VariableRow {...baseProps} profile={{ ...profile, dtype: 'Text', text_stats: {
      avg_length: 0, min_length: 0, max_length: 20,
      sentiment_distribution: { positive: 0.125, negative: 0 },
      common_words: [{ word: '', value: 'not-used', count: 1 }, { value: 'legacy', count: 2 },
        ...Array.from({ length: 9 }, (_, index) => ({ word: `word${index}`, count: index }))],
    } }} />);
    expect(screen.getByText('Avg Length').nextSibling).toHaveTextContent('0.0 chars');
    expect(screen.getByTitle('Positive: 12.5%')).toHaveStyle({ width: '12.5%' });
    expect(screen.getByText('Neu: 0%')).toBeInTheDocument();
    expect(screen.getByText('legacy (2)')).toBeInTheDocument();
    expect(screen.queryByText(/not-used/)).not.toBeInTheDocument();
    expect(screen.queryByText('word8 (8)')).not.toBeInTheDocument();
  });

  it('uses category name only when value is absent and ignores numeric bars without bins', () => {
    const handleAddFilter = vi.fn();
    const { rerender } = render(<VariableRow {...baseProps} profile={{ ...profile, dtype: 'Categorical' }} handleAddFilter={handleAddFilter} />);
    const datum = { name: 'fallback', fullName: 'full', count: 1 };
    act(() => chart.click?.(datum));
    expect(handleAddFilter).toHaveBeenCalledWith('age', 'fallback', '==');
    handleAddFilter.mockClear();
    rerender(<VariableRow {...baseProps} handleAddFilter={handleAddFilter} />);
    act(() => chart.click?.(datum));
    expect(handleAddFilter).not.toHaveBeenCalled();
  });

  it('exports the current chart with theme, filename and retained completion state', async () => {
    vi.useFakeTimers();
    vi.mocked(toPng).mockResolvedValue('data:image/png;base64,test');
    document.documentElement.classList.add('dark');
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const { rerender } = render(<VariableRow {...baseProps} />);
    await act(async () => fireEvent.click(screen.getByTitle('Download Chart')));
    expect(toPng).toHaveBeenCalledWith(screen.getByTestId('distribution').parentElement, { backgroundColor: '#1f2937', pixelRatio: 2 });
    expect(click.mock.instances[0]).toHaveProperty('download', 'age_distribution.png');
    rerender(<VariableRow {...baseProps} isExpanded={false} />);
    rerender(<VariableRow {...baseProps} />);
    expect(screen.getByTitle('Download Chart')).toBeDisabled();
    act(() => vi.advanceTimersByTime(1200));
    expect(screen.getByTitle('Download Chart')).toBeEnabled();
  });

  it('reports export failure and retains the same completion delay', async () => {
    vi.useFakeTimers();
    const error = new Error('export unavailable');
    vi.mocked(toPng).mockRejectedValue(error);
    vi.spyOn(console, 'error').mockImplementation(() => {});
    render(<VariableRow {...baseProps} />);
    await act(async () => fireEvent.click(screen.getByTitle('Download Chart')));
    expect(toPng).toHaveBeenCalledWith(screen.getByTestId('distribution').parentElement, { backgroundColor: '#ffffff', pixelRatio: 2 });
    expect(toast.error).toHaveBeenCalledWith('Chart download failed', String(error));
    expect(screen.getByTitle('Download Chart')).toBeDisabled();
    act(() => vi.advanceTimersByTime(1200));
    expect(screen.getByTitle('Download Chart')).toBeEnabled();
  });
});
