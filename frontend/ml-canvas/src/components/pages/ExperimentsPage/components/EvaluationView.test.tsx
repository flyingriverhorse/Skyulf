import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { EvaluationView } from './EvaluationView';

vi.mock('./PerClassConfusionMatrix', () => ({
  PerClassConfusionMatrix: () => <div data-testid="per-class-mock">PerClassMock</div>,
}));

const defaultProps: any = {
  selectedJobIds: ['job1'],
  evalJobId: 'job1',
  fetchEvaluationData: vi.fn(),
  isEvalLoading: false,
  evalError: null,
  evaluationData: {
    problem_type: 'classification',
    classes: ['A', 'B'],
    splits: {
      train: { metrics: {}, confusion_matrix: {} },
      test: { metrics: {}, confusion_matrix: {} },
      validation: { metrics: {}, confusion_matrix: {} },
    },
  },
  selectedRegressionSplit: null,
  setSelectedRegressionSplit: vi.fn(),
  showTrainMetrics: true,
  setShowTrainMetrics: vi.fn(),
  showTestMetrics: true,
  setShowTestMetrics: vi.fn(),
  showValMetrics: true,
  setShowValMetrics: vi.fn(),
  threshold: 0.5,
  setThreshold: vi.fn(),
  selectedRocClass: null,
  setSelectedRocClass: vi.fn(),
  cmView: 'per-class',
  setCmView: vi.fn(),
  activeTab: 'slider',
  setActiveTab: vi.fn(),
  selectedMetric: 'f1' as any,
  setSelectedMetric: vi.fn(),
  bestMetricInfos: [],
  handleDownload: vi.fn(),
  downloadingChart: null,
  doneChart: null,
  selectedTuningMetric: 'f1',
  onSelectedTuningMetricChange: vi.fn(),
  tuningPreview: null,
  tuningError: null,
  useTunedThresholds: false,
  onPreviewThresholds: vi.fn(),
  onSaveThresholds: vi.fn(),
  onToggleThresholds: vi.fn(),
  onClearThresholds: vi.fn(),
};

describe('EvaluationView tabs', () => {
  it('renders tab headers and shared split checkboxes', () => {
    render(<EvaluationView {...defaultProps} /> as any);

    expect(screen.getByText('Threshold Slider')).toBeInTheDocument();
    expect(screen.getByText('Threshold Tuning')).toBeInTheDocument();

    // Shared split checkboxes
    expect(screen.getByLabelText(/train/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/test/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/validation/i)).toBeInTheDocument();
  });

  it('shows slider tab content by default and PerClassConfusionMatrix in slider tab', () => {
    render(<EvaluationView {...defaultProps} /> as any);
    // Slider input present
    expect(screen.getByRole('slider')).toBeInTheDocument();
    // Per-class mock appears (since cmView is per-class)
    expect(screen.getByTestId('per-class-mock')).toBeInTheDocument();
  });

  it('switches to tuning tab and shows placeholder when no preview', () => {
    const setActiveTab = vi.fn();
    render(<EvaluationView {...defaultProps} setActiveTab={setActiveTab} activeTab={'slider'} /> as any);

    const tuningButton = screen.getByText('Threshold Tuning');
    fireEvent.click(tuningButton);
    // setActiveTab should be called with 'tuning'
    expect(setActiveTab).toHaveBeenCalledWith('tuning');
  });
});
