// Tests for the Threshold Slider / Threshold Tuning tab split in
// EvaluationView: verifies the two tabs show/hide the right controls, the
// Train/Test/Validation checkboxes are shared (rendered regardless of the
// active tab), and Tab 2 shows a placeholder until a preview exists.

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { EvaluationView } from './EvaluationView';
import type { EvaluationData } from '../types';
import type { ThresholdPreviewResult } from '../../../../core/api/thresholdTuning';

const evaluationData: Extract<EvaluationData, { problem_type: 'classification' | 'regression' }> = {
  problem_type: 'classification',
  splits: {
    train: {
      y_true: ['a', 'b', 'c', 'a'],
      y_pred: ['a', 'b', 'c', 'a'],
      y_proba: {
        classes: ['a', 'b', 'c'],
        values: [
          [0.7, 0.2, 0.1],
          [0.2, 0.6, 0.2],
          [0.1, 0.2, 0.7],
          [0.6, 0.3, 0.1],
        ],
      },
    },
  },
};

const noop = async () => {};

function baseProps(overrides: Partial<React.ComponentProps<typeof EvaluationView>> = {}) {
  return {
    eligibleJobIds: ['job-1'],
    evalJobId: 'job-1',
    fetchEvaluationData: noop,
    isEvalLoading: false,
    evalError: null,
    evaluationData,
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
    selectedRocClass: 'a',
    setSelectedRocClass: vi.fn(),
    cmView: 'overall' as const,
    setCmView: vi.fn(),
    activeTab: 'slider' as const,
    setActiveTab: vi.fn(),
    selectedMetric: 'f1_weighted' as const,
    setSelectedMetric: vi.fn(),
    bestMetricInfos: [],
    handleDownload: noop,
    downloadingChart: null,
    doneChart: null,
    selectedTuningMetric: 'f1',
    onSelectedTuningMetricChange: vi.fn(),
    tuningPreview: null as ThresholdPreviewResult | null,
    tuningError: null,
    useTunedThresholds: false,
    onPreviewThresholds: noop,
    onSaveThresholds: noop,
    onToggleThresholds: noop,
    onClearThresholds: noop,
    ...overrides,
  };
}

describe('EvaluationView — Threshold Slider / Threshold Tuning tabs', () => {
  it('renders both tab buttons', () => {
    render(<EvaluationView {...baseProps()} />);
    expect(screen.getByText('Threshold Slider')).toBeInTheDocument();
    expect(screen.getByText('Threshold Tuning')).toBeInTheDocument();
  });

  it('shows the manual slider controls and hides the Tuning panel when activeTab is "slider"', () => {
    render(<EvaluationView {...baseProps({ activeTab: 'slider' })} />);
    expect(screen.getByText('Class:')).toBeInTheDocument();
    expect(screen.queryByText('Preview')).not.toBeInTheDocument();
  });

  it('shows the Tuning panel and hides the manual slider controls when activeTab is "tuning"', () => {
    render(<EvaluationView {...baseProps({ activeTab: 'tuning' })} />);
    expect(screen.getByText('Preview')).toBeInTheDocument();
    expect(screen.queryByText('Class:')).not.toBeInTheDocument();
  });

  it('clicking the Threshold Tuning tab button calls setActiveTab("tuning")', () => {
    const setActiveTab = vi.fn();
    render(<EvaluationView {...baseProps({ activeTab: 'slider', setActiveTab })} />);
    fireEvent.click(screen.getByText('Threshold Tuning'));
    expect(setActiveTab).toHaveBeenCalledWith('tuning');
  });

  it('renders the shared Splits: checkboxes regardless of the active tab', () => {
    const { rerender } = render(<EvaluationView {...baseProps({ activeTab: 'slider' })} />);
    expect(screen.getByText('Train')).toBeInTheDocument();
    rerender(<EvaluationView {...baseProps({ activeTab: 'tuning' })} />);
    expect(screen.getByText('Train')).toBeInTheDocument();
  });

  it('shows a placeholder in Tab 2 until a tuning preview exists', () => {
    render(<EvaluationView {...baseProps({ activeTab: 'tuning', tuningPreview: null })} />);
    expect(screen.getByText(/Click Preview above/)).toBeInTheDocument();
  });

  it('announces preview mutations as pending and disables the triggering control', async () => {
    let resolvePreview: (() => void) | undefined;
    const onPreviewThresholds = vi.fn(
      () =>
        new Promise<void>(resolve => {
          resolvePreview = () => resolve();
        }),
    );

    render(<EvaluationView {...baseProps({ activeTab: 'tuning', onPreviewThresholds })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Preview' }));

    expect(screen.getByRole('button', { name: 'Preview' })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('Previewing tuned thresholds…');

    if (resolvePreview) {
      resolvePreview();
    }
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Preview' })).not.toBeDisabled();
    });
  });

  it('disables the tuned-threshold toggle while the mutation is in flight', async () => {
    let resolveToggle: (() => void) | undefined;
    const onToggleThresholds = vi.fn(
      () =>
        new Promise<void>(resolve => {
          resolveToggle = () => resolve();
        }),
    );

    render(<EvaluationView {...baseProps({ activeTab: 'tuning', onToggleThresholds })} />);
    fireEvent.click(screen.getByRole('checkbox', { name: /use tuned thresholds at prediction time/i }));

    expect(screen.getByRole('checkbox', { name: /use tuned thresholds at prediction time/i })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('Enabling tuned thresholds…');

    if (resolveToggle) {
      resolveToggle();
    }
    await waitFor(() => {
      expect(screen.getByRole('checkbox', { name: /use tuned thresholds at prediction time/i })).not.toBeDisabled();
    });
  });

  it('surfaces clear failures inline with a scoped retry', async () => {
    const onClearThresholds = vi
      .fn()
      .mockRejectedValueOnce(new Error('clear failed'))
      .mockResolvedValueOnce(undefined);

    render(
      <EvaluationView
        {...baseProps({
          activeTab: 'tuning',
          tuningPreview: {
            thresholds: { a: 1 },
            classes: [0],
            metric: 'f1',
            split_used: 'validation',
          },
          onClearThresholds,
        })}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('clear failed');
    fireEvent.click(screen.getByRole('button', { name: /retry clear/i }));

    await waitFor(() => {
      expect(onClearThresholds).toHaveBeenCalledTimes(2);
    });
  });

  it('renders the confusion matrix in Tab 2 once a tuning preview exists', () => {
    const tuningPreview: ThresholdPreviewResult = {
      thresholds: { a: 1, b: 1, c: 1 },
      classes: [0, 1, 2],
      metric: 'f1',
      split_used: 'train',
    };
    render(<EvaluationView {...baseProps({ activeTab: 'tuning', tuningPreview })} />);
    expect(screen.queryByText(/Click Preview above/)).not.toBeInTheDocument();
    expect(screen.getByText('a vs Rest')).toBeInTheDocument();
  });
});
