import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusBadge } from './StatusBadge';

describe('StatusBadge', () => {
  it('renders the canonical label for each known status (case-insensitive)', () => {
    const cases: Array<[string, string]> = [
      ['completed', 'Completed'],
      ['COMPLETED', 'Completed'],
      ['failed', 'Failed'],
      ['running', 'Running'],
      ['pending', 'Pending'],
      ['cancelled', 'Cancelled'],
      ['idle', 'Idle'],
    ];
    for (const [input, expected] of cases) {
      const { unmount } = render(<StatusBadge status={input} />);
      expect(screen.getByText(expected)).toBeInTheDocument();
      unmount();
    }
  });

  it('falls back to "Unknown" for an unrecognised status', () => {
    render(<StatusBadge status="bogus_state" />);
    expect(screen.getByText('Unknown')).toBeInTheDocument();
  });

  it('iconOnly hides the textual label', () => {
    render(<StatusBadge status="completed" iconOnly />);
    expect(screen.queryByText('Completed')).not.toBeInTheDocument();
  });

  it('textOnly hides the icon (no SVG renders)', () => {
    const { container } = render(<StatusBadge status="completed" textOnly />);
    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(container.querySelector('svg')).toBeNull();
  });
});
