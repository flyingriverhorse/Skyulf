import { describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
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
    expect(screen.getAllByRole('tab').map((tab) => tab.textContent)).toEqual([
      'Pipeline Basics', 'Preprocessing & Leakage', 'Split & Merge',
    ]);
  });

  it('calls onClose when the close button is clicked', () => {
    const onClose = vi.fn();
    render(<HelpGuideModal isOpen onClose={onClose} />);
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  /** Existing callers must still open the general pipeline explanation. */
  it('defaults to basics and switches to the preprocessing reference', () => {
    render(<HelpGuideModal isOpen onClose={() => {}} />);
    expect(screen.getByRole('tab', { name: 'Pipeline Basics' })).toHaveAttribute('aria-selected', 'true');
    fireEvent.mouseDown(screen.getByRole('tab', { name: 'Preprocessing & Leakage' }), { button: 0, ctrlKey: false });
    expect(screen.getByRole('searchbox', { name: 'Search preprocessing nodes' })).toBeInTheDocument();
  });

  /** Entry points from a leakage warning must land directly on the relevant guide. */
  it('opens the requested tab and resets to that tab after reopening', () => {
    const { rerender } = render(<HelpGuideModal isOpen onClose={() => {}} initialTab="leakage" />);
    expect(screen.getByRole('tab', { name: 'Preprocessing & Leakage' })).toHaveAttribute('aria-selected', 'true');
    fireEvent.mouseDown(screen.getByRole('tab', { name: 'Pipeline Basics' }), { button: 0, ctrlKey: false });
    rerender(<HelpGuideModal isOpen={false} onClose={() => {}} initialTab="leakage" />);
    rerender(<HelpGuideModal isOpen onClose={() => {}} initialTab="leakage" />);
    expect(screen.getByRole('searchbox', { name: 'Search preprocessing nodes' })).toBeInTheDocument();
  });

  /** Keyboard users must reach the same guide and retain a labelled tab panel. */
  it('switches tabs with the keyboard', async () => {
    render(<HelpGuideModal isOpen onClose={() => {}} />);
    const basics = screen.getByRole('tab', { name: 'Pipeline Basics' });
    act(() => basics.focus());
    fireEvent.keyDown(basics, { key: 'ArrowRight' });
    expect(await screen.findByRole('tabpanel', { name: 'Preprocessing & Leakage' })).toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Preprocessing & Leakage' }), { key: 'End' });
    expect(await screen.findByRole('tabpanel', { name: 'Split & Merge' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Split & Merge' })).toHaveFocus();
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Split & Merge' }), { key: 'Home' });
    expect(await screen.findByRole('tabpanel', { name: 'Pipeline Basics' })).toBeInTheDocument();
  });

  /** Moving detailed explanations must leave the catalog independently reachable. */
  it('keeps split diagrams and selection guidance in their own tab', () => {
    render(<HelpGuideModal isOpen onClose={() => {}} initialTab="leakage" />);
    const leakagePanel = screen.getByRole('tabpanel', { name: 'Preprocessing & Leakage' });
    expect(within(leakagePanel).getAllByRole('article')).toHaveLength(62);
    expect(within(leakagePanel).queryByRole('table')).not.toBeInTheDocument();

    fireEvent.mouseDown(screen.getByRole('tab', { name: 'Split & Merge' }), { button: 0, ctrlKey: false });
    const splitPanel = screen.getByRole('tabpanel', { name: 'Split & Merge' });
    expect(within(splitPanel).getByText('Train-Test Split', { selector: 'figcaption' })).toBeInTheDocument();
    expect(within(splitPanel).getByText('Feature-Target Split', { selector: 'figcaption' })).toBeInTheDocument();
    expect(within(splitPanel).getByRole('table')).toBeInTheDocument();
    expect(within(splitPanel).queryByRole('searchbox')).not.toBeInTheDocument();

    fireEvent.mouseDown(screen.getByRole('tab', { name: 'Preprocessing & Leakage' }), { button: 0, ctrlKey: false });
    expect(screen.getByRole('searchbox', { name: 'Search preprocessing nodes' })).toBeInTheDocument();
    expect(screen.getAllByRole('article')).toHaveLength(62);
  });
});
