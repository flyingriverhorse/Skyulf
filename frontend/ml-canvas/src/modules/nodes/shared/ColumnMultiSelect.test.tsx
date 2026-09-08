import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ColumnMultiSelect } from './ColumnMultiSelect';

describe('ColumnMultiSelect accessible controls', () => {
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
