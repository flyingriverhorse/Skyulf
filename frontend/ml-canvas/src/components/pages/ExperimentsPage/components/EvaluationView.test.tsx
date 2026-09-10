// Tests for the Threshold Slider / Threshold Tuning tab split in
// EvaluationView: verifies the two tabs show/hide the right controls, the
// Train/Test/Validation checkboxes are shared (rendered regardless of the
// active tab), and Tab 2 shows a placeholder until a preview exists.

import { describe, it, expect, vi } from 'vitest';
import { act, render, screen, fireEvent, waitFor } from '@testing-library/react';
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
    hasSavedThresholds: false,
    useTunedThresholds: false,
    onPreviewThresholds: noop,
    onSaveThresholds: noop,
    onToggleThresholds: noop,
    onClearThresholds: noop,
    ...overrides,
  };
}

describe('EvaluationView — Threshold Slider / Threshold Tuning tabs', () => {
  it.each([null, {
    thresholds: { '0': 0.4, '1': 0.6 }, classes: [0, 1], metric: 'f1', split_used: 'test',
  }])('requires saved thresholds before enabling predictions (preview: %j)', (tuningPreview) => {
    /** A preview cannot satisfy the backend toggle endpoint's persistence requirement. */
    const onToggleThresholds = vi.fn();
    render(<EvaluationView {...baseProps({ activeTab: 'tuning', tuningPreview, onToggleThresholds })} />);
    const toggle = screen.getByRole('checkbox', { name: /Use tuned thresholds/ });
    expect(toggle).toBeDisabled();
    expect(toggle).not.toBeChecked();
    expect(screen.getByText('Preview thresholds, then Save to enable them for predictions.')).toBeInTheDocument();
    expect(onToggleThresholds).not.toHaveBeenCalled();
  });

  it('allows previously saved thresholds to be enabled again', async () => {
    /** Disabling a saved set must not make the user recompute or save it again. */
    const onToggleThresholds = vi.fn().mockResolvedValue(undefined);
    render(<EvaluationView {...baseProps({ activeTab: 'tuning', hasSavedThresholds: true, onToggleThresholds })} />);
    const toggle = screen.getByRole('checkbox', { name: /Use tuned thresholds/ });
    expect(toggle).toBeEnabled();
    await act(async () => { fireEvent.click(toggle); });
    expect(onToggleThresholds).toHaveBeenCalledWith(true);
  });

  it('keeps stale data and its focused controls mounted while another run loads', () => {
    /** Background loading must not reset the visible chart controls or keyboard focus. */
    const props = baseProps();
    const { rerender } = render(<EvaluationView {...props} />);
    const slider = screen.getByRole('slider');
    slider.focus();
    rerender(<EvaluationView {...props} evalJobId="job-2" isEvalLoading />);
    expect(screen.getByRole('slider')).toBe(slider);
    expect(slider).toHaveFocus();
    expect(screen.getByText('Loading evaluation data…')).toBeInTheDocument();
  });

  it('preserves a pending save across tabs and disables every mutation control', async () => {
    /** Switching to manual exploration must not forget an in-flight save. */
    let finish!: () => void;
    const onSaveThresholds = vi.fn(() => new Promise<void>(resolve => { finish = resolve; }));
    const props = baseProps({ activeTab: 'tuning', onSaveThresholds, tuningPreview: {
      thresholds: { a: 0.4 }, classes: [0], metric: 'f1', split_used: 'test', source: 'training',
    } });
    const { rerender } = render(<EvaluationView {...props} />);
    expect(screen.getByText('seeded at training')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    for (const name of ['Preview', 'Save', 'Clear']) expect(screen.getByRole('button', { name })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name: /Use tuned thresholds/ })).toBeDisabled();
    rerender(<EvaluationView {...props} activeTab="slider" />);
    rerender(<EvaluationView {...props} />);
    expect(screen.getByRole('status')).toHaveTextContent('Saving tuned thresholds…');
    await act(async () => { finish(); });
    expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled();
    expect(onSaveThresholds).toHaveBeenCalledTimes(1);
  });

  it('reports a non-Error disable failure and clears it when another action starts', async () => {
    /** Mutation-specific retry text must not linger after a new successful action. */
    const onToggleThresholds = vi.fn().mockRejectedValue('failed');
    const onPreviewThresholds = vi.fn().mockResolvedValue(undefined);
    render(<EvaluationView {...baseProps({ activeTab: 'tuning', hasSavedThresholds: true, useTunedThresholds: true, onToggleThresholds, onPreviewThresholds })} />);
    fireEvent.click(screen.getByRole('checkbox', { name: /Use tuned thresholds/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Failed to disable thresholds');
    expect(onToggleThresholds).toHaveBeenCalledWith(false);
    expect(screen.getByRole('button', { name: 'Retry disable' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Preview' }));
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument());
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(onPreviewThresholds).toHaveBeenCalledTimes(1);
  });

  it('defaults regression charts to validation and honors an explicit available split', () => {
    /** Tabs and charts must derive the same fallback after the selected split disappears. */
    const split = { y_true: [1, 2, 3], y_pred: [1, 2, 3] };
    const props = baseProps({ evaluationData: { problem_type: 'regression', splits: { train: split, test: split, validation: split } }, selectedRegressionSplit: 'missing' });
    const { container, rerender } = render(<EvaluationView {...props} />);
    expect(screen.getByRole('button', { name: 'Validation' })).toHaveClass('bg-blue-500');
    expect(container.querySelector('[id*="validation"]')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Test' }));
    expect(props.setSelectedRegressionSplit).toHaveBeenCalledWith('test');
    rerender(<EvaluationView {...props} selectedRegressionSplit="test" />);
    expect(screen.getByRole('button', { name: 'Test' })).toHaveClass('bg-blue-500');
    expect(container.querySelector('[id*="test"]')).toBeInTheDocument();
    expect(container.querySelector('[id*="validation"]')).not.toBeInTheDocument();
  });

  it('forwards manual class, metric, threshold, badge and split changes', () => {
    /** Extracted controls must retain the original callback values and numeric parsing. */
    const props = baseProps({ bestMetricInfos: [{ threshold: 0.3, value: 0.9, splitLabel: 'train', metricName: 'f1_weighted' }] });
    render(<EvaluationView {...props} />);
    const [classes, metric] = screen.getAllByRole('combobox');
    fireEvent.change(classes!, { target: { value: 'b' } });
    fireEvent.change(metric!, { target: { value: 'accuracy' } });
    fireEvent.change(screen.getByRole('slider'), { target: { value: '0.27' } });
    fireEvent.click(screen.getByRole('button', { name: /train.*0.30/ }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Train' }));
    expect(props.setSelectedRocClass).toHaveBeenCalledWith('b');
    expect(props.setSelectedMetric).toHaveBeenCalledWith('accuracy');
    expect(props.setThreshold).toHaveBeenNthCalledWith(1, 0.27);
    expect(props.setThreshold).toHaveBeenNthCalledWith(2, 0.3);
    expect(props.setShowTrainMetrics).toHaveBeenCalledWith(false);
  });
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

  it('offers class-prediction metrics for threshold tuning without ROC AUC', () => {
    /** The optimizer must not promise to improve a threshold-independent ranking score. */
    const props = baseProps({ activeTab: 'tuning' });
    render(<EvaluationView {...props} />);
    expect(screen.queryByRole('option', { name: 'ROC AUC' })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: 'Threshold tuning metric' }), {
      target: { value: 'balanced_accuracy' },
    });
    expect(props.onSelectedTuningMetricChange).toHaveBeenCalledWith('balanced_accuracy');
  });

  it.each(['roc_auc', 'f1_weighted'])('requires a supported preview before resaving a saved %s set', (metric) => {
    /** An allowed dropdown value must not submit unsupported metadata from an old saved set. */
    const props = baseProps({
      activeTab: 'tuning', hasSavedThresholds: true, useTunedThresholds: true,
      tuningPreview: { thresholds: { '0': 0.4, '1': 0.6 }, classes: [0, 1], metric, split_used: 'validation' },
    });
    const { rerender } = render(<EvaluationView {...props} />);
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
    expect(screen.getByText(/click Preview before saving a replacement/)).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Use tuned thresholds at prediction time' })).toBeEnabled();
    rerender(<EvaluationView {...props} tuningPreview={{ ...props.tuningPreview!, metric: 'balanced_accuracy' }} />);
    expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled();
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

    render(<EvaluationView {...baseProps({ activeTab: 'tuning', hasSavedThresholds: true, onToggleThresholds })} />);
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

  it('renders a retry button when evaluation loading fails', () => {
    const fetchEvaluationData = vi.fn();
    render(<EvaluationView {...baseProps({ evalError: 'Failed to fetch evaluation data', fetchEvaluationData })} />);
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(fetchEvaluationData).toHaveBeenCalledTimes(1);
    expect(fetchEvaluationData).toHaveBeenCalledWith('job-1');
  });
});
