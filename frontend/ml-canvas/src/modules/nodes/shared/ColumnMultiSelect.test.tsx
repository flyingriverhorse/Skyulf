import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ColumnMultiSelect } from './ColumnMultiSelect';

describe('ColumnMultiSelect accessible controls', () => {
  it('preserves invisible selections and case-insensitive visible ordering', () => {
    // Filtering must neither lose saved hidden columns nor reorder callbacks.
    const onChange = vi.fn();
    render(<ColumnMultiSelect columns={['AGE', 'agent', 'income']} selected={['hidden', 'AGE']} onChange={onChange} />);
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Ag' } });
    expect(screen.getAllByRole('checkbox')).toHaveLength(2);
    fireEvent.click(screen.getByRole('button', { name: 'Select all matching Columns' }));
    expect(onChange).toHaveBeenLastCalledWith(['hidden', 'AGE', 'agent']);
    fireEvent.click(screen.getByRole('button', { name: 'Select none of the matching Columns' }));
    expect(onChange).toHaveBeenLastCalledWith(['hidden']);
    fireEvent.click(screen.getByRole('checkbox', { name: 'AGE' }));
    expect(onChange).toHaveBeenLastCalledWith(['hidden']);
  });

  it('keeps compact single selection, badges and footer independent of panel actions', () => {
    // Single-column operands replace even an already selected value.
    const onChange = vi.fn();
    render(<ColumnMultiSelect columns={['age']} selected={['age', 'hidden']} onChange={onChange}
      label="Operand" aria-label="Left operand" variant="compact" single showFooterCount renderItemBadge={() => <span>missing</span>} />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    expect(screen.getByText('2 selected')).toBeInTheDocument();
    expect(screen.getByText('missing')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: 'age missing' }));
    expect(onChange).toHaveBeenCalledWith(['age']);
  });

  it('retains search while loading and distinguishes empty from unmatched columns', () => {
    // Schema refreshes must retain the user's search and the proper empty message.
    const props = { columns: ['age'], selected: [], onChange: vi.fn(), fillHeight: false };
    const { rerender } = render(<ColumnMultiSelect {...props} />);
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'other' } });
    expect(screen.getByText('No columns match "other"')).toBeInTheDocument();
    rerender(<ColumnMultiSelect {...props} columns={[]} isLoading emptyMessage="Connect an input" />);
    expect(screen.getByText('Loading columns...')).toBeInTheDocument();
    rerender(<ColumnMultiSelect {...props} columns={[]} emptyMessage="Connect an input" />);
    expect(screen.getByText('Connect an input')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveValue('other');
  });
  it('names each picker and its filtered selection actions', () => {
    // Multiple column pickers must be distinguishable while preserving filtered selection behavior.
    const onChange = vi.fn();
    render(<>
      <ColumnMultiSelect columns={['age', 'income']} selected={['income']} onChange={onChange} label="Numeric Columns" />
      <ColumnMultiSelect columns={['age', 'income']} selected={[]} onChange={() => {}} label="Excluded Columns" />
    </>);
    const picker = within(screen.getByRole('group', { name: 'Numeric Columns' }));
    fireEvent.change(picker.getByRole('textbox', { name: 'Search Numeric Columns' }), { target: { value: 'age' } });
    fireEvent.click(picker.getByRole('button', { name: 'Select all matching Numeric Columns' }));
    expect(onChange).toHaveBeenLastCalledWith(['income', 'age']);
    fireEvent.click(picker.getByRole('button', { name: 'Select none of the matching Numeric Columns' }));
    expect(onChange).toHaveBeenLastCalledWith(['income']);
    expect(screen.getByRole('textbox', { name: 'Search Excluded Columns' })).toBeInTheDocument();
  });
});
