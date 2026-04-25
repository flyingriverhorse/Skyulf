import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { EmptyState } from './EmptyState';
import { ErrorState } from './ErrorState';
import { LoadingState } from './LoadingState';

describe('EmptyState', () => {
  it('renders the title', () => {
    render(<EmptyState title="Nothing here yet" />);
    expect(screen.getByText('Nothing here yet')).toBeInTheDocument();
  });

  it('renders the optional description and action', () => {
    render(
      <EmptyState
        title="Empty"
        description="Try uploading a dataset"
        action={<button>Upload</button>}
      />,
    );
    expect(screen.getByText('Try uploading a dataset')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Upload' })).toBeInTheDocument();
  });

  it('renders the default Inbox icon when no custom icon is supplied', () => {
    const { container } = render(<EmptyState title="Empty" />);
    // Default icon is an SVG from lucide-react.
    expect(container.querySelector('svg')).not.toBeNull();
  });

  it('uses the custom icon when supplied', () => {
    render(<EmptyState title="Empty" icon={<span data-testid="custom-icon" />} />);
    expect(screen.getByTestId('custom-icon')).toBeInTheDocument();
  });
});

describe('ErrorState', () => {
  it('renders the error message', () => {
    render(<ErrorState error="Network unreachable" />);
    expect(screen.getByText('Network unreachable')).toBeInTheDocument();
  });

  it('does NOT render the retry button when onRetry is omitted', () => {
    render(<ErrorState error="boom" />);
    expect(screen.queryByRole('button', { name: /retry/i })).toBeNull();
  });

  it('renders and wires the retry button when onRetry is supplied', async () => {
    const onRetry = vi.fn();
    const user = userEvent.setup();
    render(<ErrorState error="boom" onRetry={onRetry} />);
    const btn = screen.getByRole('button', { name: /retry/i });
    await user.click(btn);
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});

describe('LoadingState', () => {
  it('renders the default "Loading..." message when none provided', () => {
    render(<LoadingState />);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('renders the custom message when provided', () => {
    render(<LoadingState message="Crunching numbers" />);
    expect(screen.getByText('Crunching numbers')).toBeInTheDocument();
  });

  it('renders a spinner SVG', () => {
    const { container } = render(<LoadingState />);
    expect(container.querySelector('svg')).not.toBeNull();
  });
});
