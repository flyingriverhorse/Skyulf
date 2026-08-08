import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { EDASidebar } from './EDASidebar';
import type { EDAProfile } from '../../core/types/edaProfile';

const profile = { columns: {} } as unknown as EDAProfile;

const baseProps = {
  activeTab: 'dashboard',
  setActiveTab: () => {},
  profile,
  filtersDraft: [],
  filtersApplied: [],
  filtersDirty: false,
  columns: ['age', 'income'],
  excludedCols: [],
  excludedDirty: false,
  analyzing: false,
  onAddFilter: () => {},
  onRemoveFilter: () => {},
  onResetFilters: () => {},
  onApplyFilters: () => {},
  onToggleExclude: () => {},
  onApplyExcluded: () => {},
};

describe('EDASidebar accessible names', () => {
  it('names the filter column, operator, and value controls', () => {
    render(<EDASidebar {...baseProps} />);

    fireEvent.click(screen.getByRole('button', { name: /Add Filter/i }));

    expect(screen.getByRole('combobox', { name: 'Filter column' })).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Filter operator' })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Filter value' })).toBeInTheDocument();
  });

  it('names the exclusion column picker', () => {
    render(<EDASidebar {...baseProps} />);

    fireEvent.click(screen.getByRole('button', { name: /Excluded \(/i }));
    fireEvent.click(screen.getByRole('button', { name: /Exclude Column/i }));

    expect(
      screen.getByRole('combobox', { name: 'Column to exclude from analysis' }),
    ).toBeInTheDocument();
  });

  it('links invalid numeric filter values to the value control', () => {
    render(<EDASidebar {...baseProps} />);

    fireEvent.click(screen.getByRole('button', { name: /Add Filter/i }));
    fireEvent.change(screen.getByRole('combobox', { name: 'Filter column' }), {
      target: { value: 'age' },
    });
    fireEvent.change(screen.getByRole('combobox', { name: 'Filter operator' }), {
      target: { value: '>' },
    });
    fireEvent.change(screen.getByRole('textbox', { name: 'Filter value' }), {
      target: { value: 'abc' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Save draft/i }));

    expect(screen.getByRole('textbox', { name: 'Filter value' })).toHaveAttribute(
      'aria-invalid',
      'true',
    );
    expect(screen.getByRole('textbox', { name: 'Filter value' })).toHaveAccessibleDescription(
      /numeric value/i,
    );
  });
});

describe('EDASidebar filter apply gating', () => {
  it('disables Apply filters when the draft matches what is already applied', () => {
    render(<EDASidebar {...baseProps} filtersDirty={false} />);

    expect(screen.getByRole('button', { name: /Apply filters/i })).toBeDisabled();
  });

  it('disables Apply filters while an analysis request is in flight', () => {
    render(<EDASidebar {...baseProps} filtersDirty analyzing />);

    expect(screen.getByRole('button', { name: /Apply filters/i })).toBeDisabled();
  });

  it('enables Apply filters once the draft diverges and nothing is pending', () => {
    render(<EDASidebar {...baseProps} filtersDirty />);

    expect(screen.getByRole('button', { name: /Apply filters/i })).toBeEnabled();
  });
});
