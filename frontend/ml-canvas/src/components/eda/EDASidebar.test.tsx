import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { EDASidebar } from './EDASidebar';

const baseProps = {
  activeTab: 'dashboard', setActiveTab: vi.fn(),
  profile: { columns: {}, row_count: 0, column_count: 0 },
  filtersDraft: [], filtersApplied: [], filtersDirty: false,
  columns: ['age', 'income'], excludedCols: ['income'], excludedDirty: false,
  analyzing: false, onAddFilter: vi.fn(), onRemoveFilter: vi.fn(),
  onResetFilters: vi.fn(), onApplyFilters: vi.fn(), onToggleExclude: vi.fn(),
  onApplyExcluded: vi.fn(),
};

function fillFilter(operator: string, value: string) {
  fireEvent.click(screen.getByRole('button', { name: /Add Filter/i }));
  fireEvent.change(screen.getByRole('combobox', { name: 'Filter column' }), { target: { value: 'age' } });
  fireEvent.change(screen.getByRole('combobox', { name: 'Filter operator' }), { target: { value: operator } });
  fireEvent.change(screen.getByRole('textbox', { name: 'Filter value' }), { target: { value } });
}

describe('EDASidebar filter contracts', () => {
  it.each([
    ['==', ' 0 ', 0], ['!=', '001', 1], ['==', 'abc', 'abc'],
    ['==', 'Infinity', Infinity], ['>=', '-2.5', -2.5], ['<=', '0', 0],
  ])('keeps %s parsing of %s', (operator, value, expected) => {
    const onAddFilter = vi.fn();
    render(<EDASidebar {...baseProps} onAddFilter={onAddFilter} />);
    fillFilter(operator, value);
    fireEvent.keyDown(screen.getByRole('textbox', { name: 'Filter value' }), { key: 'Enter' });
    expect(onAddFilter).toHaveBeenCalledWith('age', expected, operator);
    fireEvent.click(screen.getByRole('button', { name: /Add Filter/i }));
    expect(screen.getByRole('textbox', { name: 'Filter value' })).toHaveValue('');
    expect(screen.getByRole('combobox', { name: 'Filter operator' })).toHaveValue('==');
  });

  it.each([['>', 'Infinity'], ['<', 'NaN'], ['>=', 'abc'], ['<=', 'abc'], ['==', '  ']])(
    'rejects invalid %s value %s after attempted submission', (operator, value) => {
      const onAddFilter = vi.fn();
      render(<EDASidebar {...baseProps} onAddFilter={onAddFilter} />);
      fillFilter(operator, value);
      expect(screen.getByRole('textbox', { name: 'Filter value' })).not.toHaveAttribute('aria-invalid', 'true');
      fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
      expect(onAddFilter).not.toHaveBeenCalled();
      expect(screen.getByRole('textbox', { name: 'Filter value' })).toHaveAttribute('aria-invalid', 'true');
    },
  );

  it('retains draft and section state through collapsing, PII navigation and parent updates', () => {
    const { rerender } = render(<EDASidebar {...baseProps} />);
    fillFilter('!=', 'draft');
    const input = screen.getByRole('textbox', { name: 'Filter value' });
    input.focus();
    fireEvent.change(input, { target: { value: 'editing' } });
    expect(screen.getByRole('textbox', { name: 'Filter value' })).toBe(input);
    expect(input).toHaveFocus();
    fireEvent.click(screen.getByRole('button', { name: 'Collapse Sidebar' }));
    fireEvent.click(screen.getByRole('button', { name: 'Expand Sidebar' }));
    expect(screen.getByRole('textbox', { name: 'Filter value' })).toHaveValue('editing');
    rerender(<EDASidebar {...baseProps} activeTab="pii" />);
    expect(screen.queryByRole('textbox', { name: 'Filter value' })).not.toBeInTheDocument();
    rerender(<EDASidebar {...baseProps} />);
    expect(screen.getByRole('textbox', { name: 'Filter value' })).toHaveValue('editing');
    expect(screen.getByRole('combobox', { name: 'Filter operator' })).toHaveValue('!=');
  });

  it('detects ordered and typed signature differences even without the dirty flag', () => {
    const first = { column: 'age', operator: '==', value: 1 };
    const second = { column: 'income', operator: '==', value: [1, '2'] };
    const { rerender } = render(<EDASidebar {...baseProps} filtersDraft={[first, second]} filtersApplied={[first, second]} />);
    expect(screen.getByRole('button', { name: 'Apply filters' })).toBeDisabled();
    rerender(<EDASidebar {...baseProps} filtersDraft={[second, first]} filtersApplied={[first, second]} />);
    expect(screen.getByRole('button', { name: 'Apply filters' })).toBeEnabled();
    expect(screen.queryByText('Pending')).not.toBeInTheDocument();
    rerender(<EDASidebar {...baseProps} filtersDraft={[{ ...first, value: '1' }]} filtersApplied={[first]} />);
    expect(screen.getByRole('button', { name: 'Reset filters' })).toBeEnabled();
  });

  it('limits exclusion choices and preserves its form while collapsed', () => {
    const onToggleExclude = vi.fn();
    render(<EDASidebar {...baseProps} onToggleExclude={onToggleExclude} excludedDirty analyzing />);
    fireEvent.click(screen.getByRole('button', { name: /Excluded \(/ }));
    expect(screen.getByRole('button', { name: 'Apply changes' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Exclude Column' }));
    fireEvent.click(screen.getByRole('button', { name: 'Collapse Sidebar' }));
    fireEvent.click(screen.getByRole('button', { name: 'Expand Sidebar' }));
    expect(screen.queryByRole('option', { name: 'income' })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: 'Column to exclude from analysis' }), { target: { value: 'age' } });
    expect(onToggleExclude).toHaveBeenCalledWith('age', true);
  });

  it('keeps optional navigation visibility, collapsed names and active page semantics', () => {
    const setActiveTab = vi.fn();
    const { rerender } = render(<EDASidebar {...baseProps} setActiveTab={setActiveTab} />);
    expect(screen.queryByRole('button', { name: 'Sample Data' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Target Analysis' })).not.toBeInTheDocument();
    expect(screen.queryByText('Specialized')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Decomposition' })).toBeInTheDocument();
    rerender(<EDASidebar {...baseProps} setActiveTab={setActiveTab} profile={{ ...baseProps.profile,
      sample_data: [], target_col: 'age', target_correlations: {}, correlations_with_target: {}, clustering: {}, timeseries: {},
    }} />);
    expect(screen.getByRole('button', { name: 'Sample Data' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Target Analysis' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Correlations' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'PCA & Clusters' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Collapse Sidebar' }));
    expect(screen.getByRole('button', { name: 'Dashboard' })).toHaveAttribute('aria-current', 'page');
    expect(screen.getByRole('button', { name: 'Time Series' })).toHaveAttribute('title', 'Time Series');
    fireEvent.click(screen.getByRole('button', { name: 'Time Series' }));
    expect(setActiveTab).toHaveBeenCalledWith('timeseries');
  });
});
