import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { CausalTab } from './CausalTab';
import { CorrelationsTab } from './CorrelationsTab';
import { AnalysisNavigation } from '../edaSidebar/AnalysisNavigation';

const graph = {
  nodes: [{ id: 'measurement', label: 'measurement' }, { id: 'outcome', label: 'outcome' }],
  edges: [{ source: 'measurement', target: 'outcome', type: 'directed' }],
};

describe('causal report explanations', () => {
  it.each([
    ['all', /all eligible numeric variables/i],
    ['target_correlation', /selected numeric target.*14.*correlat/i],
    ['variance', /15.*highest.variance numeric variables/i],
    [undefined, /up to 15 numeric variables/i],
  ])('describes selection %s without promising an absent target', (selection_method, message) => {
    /** Selection metadata determines the explanation, including a cautious fallback for cached reports. */
    render(<CausalTab profile={{ causal_graph: { ...graph, selection_method }, target_col: 'outcome' }} />);
    expect(screen.getByText(message)).toBeInTheDocument();
  });

  it.each([
    ['categorical', /species.*categorical.*omitted/i],
    ['excluded', /species.*excluded/i],
    ['unsupported', /species.*not supported/i],
  ])('explains the %s target even when discovery returns no graph', (reason, message) => {
    /** Omitted-target metadata remains useful when there are insufficient numeric variables for a graph. */
    render(<CausalTab profile={{ causal_graph: null, target_col: 'species', causal_target_exclusion_reason: reason }} />);
    expect(screen.getByText(/no causal graph available/i)).toBeInTheDocument();
    expect(screen.getByText(message)).toBeInTheDocument();
  });

  it('uses categorical metadata for a numeric classification target without guessing from dtype', () => {
    /** Numeric class identifiers are categories too and must not be advertised as numeric causal inputs. */
    render(<CausalTab profile={{
      causal_graph: graph, target_col: 'class_id', causal_target_exclusion_reason: 'categorical',
      columns: { class_id: { dtype: 'Numeric' } },
    }} />);
    expect(screen.getByText(/class_id.*categorical.*omitted/i)).toBeInTheDocument();
    expect(screen.getByText(/Target Analysis.*associations/i)).toBeInTheDocument();
  });

  it.each(['Causal Graph', 'Correlations'])('keeps %s reachable when numeric analysis has no results', (tab) => {
    /** Hiding the tab would prevent users from learning why their selected target is absent. */
    render(<AnalysisNavigation profile={{
      row_count: 100, column_count: 1, columns: {},
      causal_graph: null, correlations: null, correlations_with_target: null,
      causal_target_exclusion_reason: 'categorical',
    }} activeTab="causal" setActiveTab={() => {}} isCollapsed={false} setIsCollapsed={() => {}} />);
    expect(screen.getByRole('button', { name: tab })).toBeInTheDocument();
  });

  it('explains the missing target Pearson matrix using the same eligibility metadata', () => {
    /** Categorical associations remain available instead of an arbitrary coded Pearson matrix. */
    render(<CorrelationsTab profile={{
      target_col: 'species', causal_target_exclusion_reason: 'categorical',
      correlations: { columns: ['measurement'], values: [[1]] },
    }} />);
    expect(screen.getByText(/species.*categorical.*omitted/i)).toBeInTheDocument();
    expect(screen.getByText(/Pearson/i)).toBeInTheDocument();
  });
});
