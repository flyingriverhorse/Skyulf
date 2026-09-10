import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { StrategySettingsModal } from './StrategySettingsModal';

/** Apply removes an empty timeout and calls save before close, while preserving zero. */
it.each([{ draft: '', expected: {} }, { draft: '0', expected: { timeout: 0 } }, { draft: '2.8', expected: { timeout: 2 } }])(
  'applies Optuna timeout $draft with the original payload', ({ draft, expected }) => {
    const calls: unknown[] = [];
    render(<StrategySettingsModal isOpen strategy="optuna" onSave={config => calls.push(config)} onClose={() => calls.push('close')} />);
    fireEvent.change(screen.getByLabelText('Timeout (Seconds)'), { target: { value: draft } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply Settings' }));
    expect(calls).toEqual([{ sampler: 'tpe', pruner: 'median', ...expected }, 'close']);
  },
);

/** Closing and reopening discards unsaved drafts; reset selects the current strategy defaults. */
it('keeps the draft lifetime and switches between strategy control groups', () => {
  const onSave = vi.fn();
  const onClose = vi.fn();
  const props = { onSave, onClose, strategy: 'halving_grid', initialConfig: { factor: 5 } };
  const { rerender } = render(<StrategySettingsModal {...props} isOpen />);
  expect(screen.getByLabelText('Factor')).toHaveValue(5);
  fireEvent.change(screen.getByLabelText('Factor'), { target: { value: '7' } });
  fireEvent.keyDown(window, { key: 'Escape' });
  expect(onClose).toHaveBeenCalledOnce();
  expect(onSave).not.toHaveBeenCalled();
  rerender(<StrategySettingsModal {...props} isOpen={false} />);
  rerender(<StrategySettingsModal {...props} isOpen />);
  expect(screen.getByLabelText('Factor')).toHaveValue(5);
  fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
  fireEvent.change(screen.getByLabelText('Min Resources'), { target: { value: '10' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply Settings' }));
  expect(onSave).toHaveBeenLastCalledWith({ factor: 3, min_resources: '10', resource: 'n_samples' });
  rerender(<StrategySettingsModal isOpen strategy="optuna" modelKey="logistic_regression" onSave={onSave} onClose={onClose} />);
  expect(screen.queryByLabelText('Factor')).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Sampler'), { target: { value: 'cmaes' } });
  expect(screen.getByText('Partial CMA-ES coverage.')).toBeVisible();
});

/** Nonempty initial configs are not merged with defaults on save. */
it('preserves partial initial payloads and unsupported strategy visibility', () => {
  const onSave = vi.fn();
  const props = { onSave, onClose: vi.fn(), isOpen: true, initialConfig: { factor: 0 } };
  const { rerender } = render(<StrategySettingsModal {...props} strategy="halving_random" />);
  expect(screen.getByLabelText('Factor')).toHaveValue(0);
  fireEvent.click(screen.getByRole('button', { name: 'Apply Settings' }));
  expect(onSave).toHaveBeenLastCalledWith({ factor: 0 });
  rerender(<StrategySettingsModal {...props} strategy="grid" />);
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});
