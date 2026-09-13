import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { EDAProfile } from '../../../core/types/edaProfile';
import { TargetAnalysisTab } from './TargetAnalysisTab';

describe('TargetAnalysisTab', () => {
  it('renders an unavailable association from a saved JSON profile', () => {
    /** Nonfinite analysis results serialize to null and must not crash the target tab. */
    render(<TargetAnalysisTab profile={{ columns: {}, target_col: 'group', target_correlations: { amount: null } }}
      downloadChart={vi.fn()} history={[]} loading={false} loadSpecificReport={vi.fn()} report={null} />);
    expect(screen.getByText('amount', { selector: 'strong' })).toBeInTheDocument();
    expect(screen.queryByText('(0.00)')).not.toBeInTheDocument();
  });

  it.each([null, 0, 0.03])('exports the actual ANOVA result %s', (p_value) => {
    /** An unavailable statistical test must not become a significant zero in chart downloads. */
    const profile: EDAProfile = {
      row_count: 4, column_count: 1, columns: {}, target_col: 'group',
      target_correlations: { amount: 0.5 },
      target_interactions: [{
        feature: 'amount', plot_type: 'boxplot', p_value,
        data: [{ name: 'a', stats: { min: 1, q1: 2, median: 3, q3: 4, max: 5 } }],
      }],
    };
    const downloadChart = vi.fn();
    render(<TargetAnalysisTab profile={profile} downloadChart={downloadChart}
      history={[]} loading={false} loadSpecificReport={vi.fn()} report={{ id: 1 }} />);

    fireEvent.click(screen.getAllByRole('button', { name: 'Download Chart' })[1]!);

    expect(downloadChart).toHaveBeenCalledWith(
      'interaction-chart-0', 'interaction-amount', 'amount vs group',
      p_value === null ? undefined : `ANOVA p: ${p_value.toExponential(2)}`,
    );
  });
});
