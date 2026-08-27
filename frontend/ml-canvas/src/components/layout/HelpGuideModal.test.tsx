import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { HelpGuideModal } from './HelpGuideModal';

describe('HelpGuideModal', () => {
  it('renders nothing when closed', () => {
    const { container } = render(<HelpGuideModal isOpen={false} onClose={() => {}} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders every concept section when open', () => {
    render(<HelpGuideModal isOpen onClose={() => {}} />);
    expect(screen.getByRole('dialog', { name: 'How pipelines work' })).toBeInTheDocument();
    for (const heading of [
      'Linear chain — the basics',
      'Branches — one input, many paths',
      'Merging — which branch wins?',
      'After a Split node — order decides',
      'Row alignment — branches must stay in step',
      'Preview vs running experiments',
      'Where your results live',
      'Can you trust the scores?',
      'Score Advisory — the amber tile in Jobs',
      'Badges and edge colors',
    ]) {
      expect(screen.getByRole('heading', { name: heading })).toBeInTheDocument();
    }
    // The hardest concepts get explicit copy checks.
    expect(screen.getByText(/last connected branch wins every shared column/)).toBeInTheDocument();
    expect(screen.getByText(/optimistically biased/)).toBeInTheDocument();
    expect(screen.getByText(/Leakage Gate/)).toBeInTheDocument();
  });

  it('calls onClose when the close button is clicked', () => {
    const onClose = vi.fn();
    render(<HelpGuideModal isOpen onClose={onClose} />);
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
