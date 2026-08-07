import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { EDASidebar } from './EDASidebar';
import type { EDAProfile } from '../../core/types/edaProfile';

const profile = { columns: {} } as unknown as EDAProfile;

const baseProps = {
  activeTab: 'dashboard',
  setActiveTab: () => {},
  profile,
  filters: [],
  columns: ['age', 'income'],
  excludedCols: [],
  excludedDirty: false,
  analyzing: false,
  onAddFilter: () => {},
  onRemoveFilter: () => {},
  onClearFilters: () => {},
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
});
