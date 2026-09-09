import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { ExperimentRunDialog } from './ExperimentRunDialog';

const models = [
  { id: 'classifier', name: 'Customer classifier', model: 'random forest classifier', action: 'Train' },
  { id: 'regressor', name: 'Revenue predictor', model: 'ridge regression', action: 'Tune' },
];

describe('ExperimentRunDialog', () => {
  it('reviews each model and its action before allowing submission', () => {
    // Model scope must remain visible so users can review what they are queuing.
    const onSubmit = vi.fn();
    render(<ExperimentRunDialog isOpen models={models} blockReason="" onClose={vi.fn()}
      onSubmit={onSubmit} onReviewIssues={vi.fn()} />);

    const rows = screen.getAllByRole('listitem');
    expect(rows).toHaveLength(2);
    expect(within(rows[0]!).getByText('Customer classifier')).toBeInTheDocument();
    expect(within(rows[0]!).getByText('random forest classifier')).toBeInTheDocument();
    expect(within(rows[0]!).getByText('Train')).toBeInTheDocument();
    expect(within(rows[1]!).getByText('Revenue predictor')).toBeInTheDocument();
    expect(within(rows[1]!).getByText('Tune')).toBeInTheDocument();
    const submit = screen.getByRole('button', { name: 'Queue experiments' });
    expect(submit).toBeEnabled();
    expect(submit).toHaveAccessibleDescription(/Runs training or tuning/);
    fireEvent.click(submit);
    expect(onSubmit).toHaveBeenCalledOnce();
  });

  it('blocks queuing and offers validation review when no model can run', () => {
    // The explanation must describe the disabled action for screen-reader users too.
    const onSubmit = vi.fn();
    const onReviewIssues = vi.fn();
    render(<ExperimentRunDialog isOpen models={[]} blockReason="Connect a model to a dataset pipeline first."
      onClose={vi.fn()} onSubmit={onSubmit} onReviewIssues={onReviewIssues} />);

    const submit = screen.getByRole('button', { name: 'Queue experiments' });
    expect(submit).toBeDisabled();
    expect(submit).toHaveAccessibleDescription('Connect a model to a dataset pipeline first.');
    fireEvent.click(submit);
    expect(onSubmit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Review validation issues' }));
    expect(onReviewIssues).toHaveBeenCalledOnce();
  });

  it('cancels without submitting any experiments', () => {
    // Cancelling a review must not accidentally queue expensive background work.
    const onClose = vi.fn();
    const onSubmit = vi.fn();
    render(<ExperimentRunDialog isOpen models={models} blockReason="" onClose={onClose}
      onSubmit={onSubmit} onReviewIssues={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalledOnce();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('removes review controls when the dialog is closed', () => {
    // Hidden confirmation controls must not remain available to keyboard users.
    render(<ExperimentRunDialog isOpen={false} models={models} blockReason=""
      onClose={vi.fn()} onSubmit={vi.fn()} onReviewIssues={vi.fn()} />);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Queue experiments' })).not.toBeInTheDocument();
  });
});
