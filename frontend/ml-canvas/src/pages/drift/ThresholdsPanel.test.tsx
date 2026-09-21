import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { ThresholdsPanel } from './ThresholdsPanel';
import type { DriftReport } from '../../core/api/monitoring';

it('retains zero and unbounded positive PSI while rejecting negative and out-of-range KS', () => {
    // Overrides must follow the metric domain instead of truthiness or HTML hints.
    const onChange = vi.fn();
    render(<ThresholdsPanel thresholds={{ psi: 0.2, ks: 0.1 }} onChange={onChange} />);
    fireEvent.change(screen.getByRole('spinbutton', { name: /^PSI/ }), { target: { value: '0' } });
    expect(onChange).toHaveBeenLastCalledWith({ psi: 0, ks: 0.1 });
    fireEvent.change(screen.getByRole('spinbutton', { name: /^PSI/ }), { target: { value: '5' } });
    expect(onChange).toHaveBeenLastCalledWith({ psi: 5, ks: 0.1 });
    onChange.mockClear();
    fireEvent.change(screen.getByRole('spinbutton', { name: /^PSI/ }), { target: { value: '-1' } });
    fireEvent.change(screen.getByRole('spinbutton', { name: /^KS statistic/ }), { target: { value: '1.1' } });
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getAllByRole('alert')).toHaveLength(2);
});

it('distinguishes clearing overrides from applying defaults', () => {
    // Clearing is a saved-report fallback, not an invisible replacement by defaults.
    const onChange = vi.fn();
    render(<ThresholdsPanel thresholds={{ psi: 0.5 }} onChange={onChange} />);
    fireEvent.change(screen.getByRole('spinbutton', { name: /^PSI/ }), { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith({ psi: undefined });
    fireEvent.click(screen.getByRole('button', { name: 'Remove overrides' }));
    expect(onChange).toHaveBeenLastCalledWith({});
    fireEvent.click(screen.getByRole('button', { name: 'Reset defaults' }));
    expect(onChange).toHaveBeenLastCalledWith({ psi: 0.2, ks: 0.1, wasserstein: 0.1, kl: 0.1 });
});

it('shows the actual saved fallback independently from the default and current override', () => {
    // A cleared field must expose the report threshold used by the metric verdict.
    const report: DriftReport = {
        reference_rows: 2, current_rows: 2, drifted_columns_count: 0, severity: 'none',
        missing_columns: [], new_columns: [], column_drifts: {
            category: { column: 'category', drift_detected: false, suggestions: [],
                metrics: [{ metric: 'psi_categorical', value: 0.5, threshold: 0.7, has_drift: false }] },
        },
    };
    const { rerender } = render(<ThresholdsPanel thresholds={{}} onChange={vi.fn()} report={report} />);
    expect(screen.getByText('Effective PSI: 0.7 (saved report)')).toBeVisible();
    expect(screen.getByText('Effective KS statistic: 0.1 (default)')).toBeVisible();
    rerender(<ThresholdsPanel thresholds={{ psi: 0 }} onChange={vi.fn()} report={report} />);
    expect(screen.getByText('Effective PSI: 0 (override)')).toBeVisible();
});
