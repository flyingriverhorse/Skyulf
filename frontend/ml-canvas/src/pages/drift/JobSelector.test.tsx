import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { JobSelector } from './JobSelector';

describe('JobSelector public selection', () => {
    it('searches metadata, selects, restores Escape focus and clears without reopening', () => {
        /** Reference selection must retain metadata search and keyboard behavior. */
        const onSelect = vi.fn();
        const jobs = [{ job_id: 'job-123456789', dataset_name: 'Sales', filename: 'sales.csv', created_at: '2026-09-10 10:00', model_type: 'forest', target_column: 'profit', description: 'seasonal', best_metric: 'R2: 0.9 | MAE: 1' }];
        const { rerender } = render(<JobSelector jobs={jobs} selectedJob="" onSelect={onSelect} invalid />);
        const trigger = screen.getByRole('button', { name: /select reference/i });
        expect(screen.getByText('Reference job is required.')).toBeInTheDocument();
        fireEvent.click(trigger);
        fireEvent.change(screen.getByPlaceholderText('Search jobs...'), { target: { value: 'PROFIT' } });
        const option = screen.getByRole('option');
        expect(option).toHaveTextContent('target: profit');
        expect(option).toHaveTextContent('R2: 0.9');
        fireEvent.click(option);
        expect(onSelect).toHaveBeenCalledWith(jobs[0]!.job_id);
        rerender(<JobSelector jobs={jobs} selectedJob={jobs[0]!.job_id} onSelect={onSelect} />);
        fireEvent.click(trigger);
        expect(screen.getByPlaceholderText('Search jobs...')).toHaveValue('');
        fireEvent.keyDown(document, { key: 'Escape' });
        expect(trigger).toHaveFocus();
        fireEvent.keyDown(screen.getByRole('button', { name: 'Clear selected reference model' }), { key: ' ' });
        expect(onSelect).toHaveBeenLastCalledWith('');
        expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    });
});
