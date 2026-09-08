import { describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { HelpGuideModal } from './HelpGuideModal';

describe('PreprocessingPlacementGuide', () => {
  /** Saved graphs can contain aliases, so every registered ID must be discoverable. */
  it('shows all 62 registrations including distinct aliases', () => {
    render(<HelpGuideModal isOpen onClose={() => {}} initialTab="leakage" />);
    expect(screen.getAllByRole('article')).toHaveLength(62);
    for (const id of ['FeatureGeneration', 'FeatureGenerationNode', 'FeatureMath', 'TrainTestSplitter', 'Split']) {
      expect(screen.getByText(id, { selector: 'code' })).toBeInTheDocument();
    }
  });

  /** A pasted registry ID should find its operation guidance despite case and whitespace. */
  it('searches registry IDs without case or surrounding whitespace sensitivity', () => {
    render(<HelpGuideModal isOpen onClose={() => {}} initialTab="leakage" />);
    fireEvent.change(screen.getByRole('searchbox', { name: 'Search preprocessing nodes' }), {
      target: { value: '  fEaTuReMaTh  ' },
    });
    expect(screen.getAllByRole('article')).toHaveLength(1);
    expect(screen.getByText('FeatureMath', { selector: 'code' })).toBeInTheDocument();
  });

  /** Users need to find mixed-operation nodes from an operation name alone. */
  it('searches operation descriptions across feature-generation aliases', () => {
    render(<HelpGuideModal isOpen onClose={() => {}} initialTab="leakage" />);
    fireEvent.change(screen.getByRole('searchbox', { name: 'Search preprocessing nodes' }), {
      target: { value: 'group_agg' },
    });
    expect(screen.getAllByRole('article')).toHaveLength(3);
    expect(screen.getByText('FeatureGenerationNode', { selector: 'code' })).toBeInTheDocument();
  });

  /** Temporal transforms need a distinct route through the catalog despite not fitting state. */
  it('filters by temporal placement and combines the filter with search', () => {
    render(<HelpGuideModal isOpen onClose={() => {}} initialTab="leakage" />);
    fireEvent.change(screen.getByRole('combobox', { name: 'Filter by placement' }), {
      target: { value: 'time' },
    });
    expect(screen.getAllByRole('article')).toHaveLength(2);
    fireEvent.change(screen.getByRole('searchbox', { name: 'Search preprocessing nodes' }), {
      target: { value: 'lag' },
    });
    expect(screen.getAllByRole('article')).toHaveLength(1);
    expect(screen.getByText('LagFeatures', { selector: 'code' })).toBeInTheDocument();
  });

  /** Clearing an unmatched query must recover the catalog without reopening the modal. */
  it('provides an empty state and a reset action', () => {
    render(<HelpGuideModal isOpen onClose={() => {}} initialTab="leakage" />);
    fireEvent.change(screen.getByRole('searchbox', { name: 'Search preprocessing nodes' }), {
      target: { value: 'does-not-exist-777' },
    });
    expect(screen.queryAllByRole('article')).toHaveLength(0);
    fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }));
    expect(screen.getAllByRole('article')).toHaveLength(62);
  });
});

/** Both saved polynomial aliases must expose their operation-sensitive placement in the catalog. */
it('keeps both polynomial aliases searchable under conditional placement', () => {
  render(<HelpGuideModal isOpen={true} onClose={() => undefined} initialTab="leakage" />);
  fireEvent.change(screen.getByRole('searchbox', { name: 'Search preprocessing nodes' }), {
    target: { value: 'PolynomialFeatures' },
  });
  fireEvent.change(screen.getByRole('combobox', { name: 'Filter by placement' }), {
    target: { value: 'conditional' },
  });
  const entries = screen.getAllByRole('article');
  expect(entries).toHaveLength(2);
  for (const entry of entries) {
    expect(entry.textContent).toContain('auto_detect');
    expect(entry.textContent).toContain('Depends on operation');
  }
  fireEvent.change(screen.getByRole('combobox', { name: 'Filter by placement' }), {
    target: { value: 'before' },
  });
  expect(screen.queryAllByRole('article')).toHaveLength(0);
});
