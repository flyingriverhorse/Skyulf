import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { SegmentationView } from './SegmentationView';

describe('SegmentationView error retry', () => {
  it('renders a retry button when evaluation loading fails', () => {
    const fetchEvaluationData = vi.fn();
    render(
      <SegmentationView
        selectedJobIds={['job-1']}
        evalJobId="job-1"
        fetchEvaluationData={fetchEvaluationData}
        isEvalLoading={false}
        evalError="Failed to fetch evaluation data"
        evaluationData={null}
        handleDownload={vi.fn()}
        downloadingChart={null}
        doneChart={null}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(fetchEvaluationData).toHaveBeenCalledTimes(1);
    expect(fetchEvaluationData).toHaveBeenCalledWith('job-1');
  });
});
