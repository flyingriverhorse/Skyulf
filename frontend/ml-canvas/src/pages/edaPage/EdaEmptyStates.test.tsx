import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { EdaReportContent } from './EdaReportContent';
import type { EdaPageModel } from './useEdaPageController';

vi.mock('../../core/utils/chartUtils', () => ({ downloadChart: vi.fn() }));
vi.mock('../../components/eda/ThreeDScatterPlot', () => ({ ThreeDScatterPlot: () => null }));

function model(activeTab: string): EdaPageModel {
  // A saved view can retain its selected tab when a new report lacks that section.
  const profile = { row_count: 4, column_count: 0, columns: {}, alerts: [], missing_cells_percentage: 0, duplicate_rows: 0 };
  return { activeTab, report: { id: 1, status: 'COMPLETED', profile_data: profile }, profileForUi: profile,
    filtersDraft: [], filtersApplied: [], excludedColsDraft: [], excludedColsApplied: [],
    history: [], scatter: {}, setActiveTab: vi.fn(), setScatter: vi.fn(), runAnalysis: vi.fn(),
  } as unknown as EdaPageModel;
}

it.each(['geospatial', 'target', 'timeseries', 'outliers', 'correlations', 'rules', 'sample', 'decomposition'])(
  'explains missing results when the selected %s section is absent', activeTab => {
    // No recorded reason means unavailable, not an invented prerequisite or algorithm failure.
    render(<EdaReportContent {...model(activeTab)} />);
    expect(screen.getByText('Analysis results unavailable')).toBeVisible();
    expect(screen.getByText(/No reason was recorded/)).toBeVisible();
  },
);

it('shows a recorded report failure and retries instead of calling it an empty result', () => {
  // Persisted failure evidence must remain distinguishable from missing optional sections.
  const props = model('outliers');
  props.report = { ...props.report!, status: 'FAILED', error_message: 'Recorded analysis failure' };
  render(<EdaReportContent {...props} />);
  expect(screen.getByText('Analysis Failed')).toBeVisible();
  expect(screen.getByText('Recorded analysis failure')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
  expect(props.runAnalysis).toHaveBeenCalledOnce();
});
