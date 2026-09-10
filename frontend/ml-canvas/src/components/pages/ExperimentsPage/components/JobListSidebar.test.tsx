import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import type { JobInfo } from '../../../../core/api/jobs';
import { JobListSidebar } from './JobListSidebar';

const job: JobInfo = { job_id: 'job-1', pipeline_id: 'preview_abcdefghijk__branch_0', node_id: 'node', job_type: 'tuning', status: 'completed', start_time: null, end_time: null, error: null, result: null, created_at: '2026-01-01', model_type: 'forest', config: { tuning: { strategy: 'legacy-search' } }, branch_index: 0, promoted_at: '2026-01-02' };
const props = () => ({ filteredJobs: [job], selectedJobIds: ['job-1'], isSidebarCollapsed: false, setIsSidebarCollapsed: vi.fn(), toggleJobSelection: vi.fn(), hasMore: true, isLoading: false, loadMoreJobs: vi.fn(), handlePromote: vi.fn((e: React.MouseEvent) => e.stopPropagation()), handleDeploy: vi.fn((e: React.MouseEvent) => e.stopPropagation()), getDuration: vi.fn(() => '3s') });

describe('job sidebar public interactions', () => {
  it('preserves legacy subtitles, winner/path badges and parent-controlled action propagation', () => {
    // Row selection must remain separate when the parent stops action bubbling.
    const handlers = props();
    render(<JobListSidebar {...handlers} />);
    expect(screen.getByText(/Unknown Dataset/)).toHaveTextContent('forest');
    expect(screen.getByText('(legacy-search)')).toBeInTheDocument();
    expect(screen.getByText('path A')).toBeInTheDocument();
    expect(screen.getByText('Winner')).toBeInTheDocument();
    fireEvent.click(screen.getByTitle('Unpromote'));
    fireEvent.click(screen.getByTitle('Deploy to Test'));
    expect(handlers.handlePromote).toHaveBeenCalledWith(expect.anything(), job);
    expect(handlers.handleDeploy).toHaveBeenCalledWith(expect.anything(), job.job_id);
    expect(handlers.toggleJobSelection).not.toHaveBeenCalled();
    fireEvent.keyDown(screen.getByRole('button', { name: /abcdefghij/ }), { key: 'Enter' });
    expect(handlers.toggleJobSelection).toHaveBeenCalledWith('job-1');
    fireEvent.click(screen.getByTitle('Load More Runs'));
    expect(handlers.loadMoreJobs).toHaveBeenCalledTimes(1);
  });

  it('preserves collapsed status, controlled expansion and filtered empty/loading states', () => {
    // Compact mode still selects runs while hiding expanded metadata and actions.
    const handlers = props();
    const { rerender } = render(<JobListSidebar {...handlers} isSidebarCollapsed isLoading />);
    expect(screen.queryByTitle('Unpromote')).not.toBeInTheDocument();
    expect(screen.getByTitle('Load More Runs')).toBeDisabled();
    fireEvent.click(screen.getByTitle('abcdefghij · forest'));
    expect(handlers.toggleJobSelection).toHaveBeenCalledWith('job-1');
    fireEvent.click(screen.getByTitle('Expand Sidebar'));
    expect(handlers.setIsSidebarCollapsed).toHaveBeenCalledWith(false);
    rerender(<JobListSidebar {...handlers} filteredJobs={[]} hasMore={false} />);
    expect(screen.getByText('No runs match the current filters.')).toBeInTheDocument();
    expect(screen.queryByTitle('Load More Runs')).not.toBeInTheDocument();
  });
});
