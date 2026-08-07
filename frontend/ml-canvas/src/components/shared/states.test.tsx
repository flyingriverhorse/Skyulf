import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { EmptyState } from './EmptyState';
import { ErrorState } from './ErrorState';
import { LoadingState } from './LoadingState';

describe('EmptyState', () => {
  it('announces the empty-state message once', () => {
    render(<EmptyState title="Nothing here yet" />);
    expect(screen.getByRole('status')).toHaveTextContent('Nothing here yet');
  });

  it('renders the optional description and action', () => {
    render(
      <EmptyState
        title="Empty"
        description="Try uploading a dataset"
        action={<button>Upload</button>}
      />,
    );
    expect(screen.getByRole('status')).toHaveTextContent('Empty');
    expect(screen.getByText('Try uploading a dataset')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Upload' })).toBeInTheDocument();
  });

  it('hides the decorative icon from assistive technology', () => {
    const { container } = render(<EmptyState title="Empty" />);
    const icon = container.querySelector('svg');
    expect(icon).not.toBeNull();
    expect(icon).toHaveAttribute('aria-hidden', 'true');
  });

  it('uses the custom icon when supplied', () => {
    render(<EmptyState title="Empty" icon={<span data-testid="custom-icon" />} />);
    expect(screen.getByTestId('custom-icon')).toBeInTheDocument();
    expect(screen.getByTestId('custom-icon').parentElement).toHaveAttribute('aria-hidden', 'true');
  });
});

describe('ErrorState', () => {
  it('announces failures as an alert', () => {
    render(<ErrorState error="Network unreachable" />);
    expect(screen.getByRole('alert')).toHaveTextContent('Network unreachable');
  });

  it('does NOT render the retry button when onRetry is omitted', () => {
    render(<ErrorState error="boom" />);
    expect(screen.queryByRole('button', { name: /retry/i })).toBeNull();
  });

  it('links the retry action to the error message', () => {
    const onRetry = vi.fn();
    render(<ErrorState error="boom" onRetry={onRetry} />);
    const btn = screen.getByRole('button', { name: /retry/i });
    expect(btn).toHaveAccessibleDescription('boom');
    fireEvent.click(btn);
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('hides every decorative icon inside the alert', () => {
    const { container } = render(<ErrorState error="boom" onRetry={vi.fn()} />);
    const exposedIcons = Array.from(container.querySelectorAll('svg')).filter(
      (icon) => icon.closest('[aria-hidden="true"]') === null,
    );
    expect(exposedIcons).toHaveLength(0);
  });
});

describe('LoadingState', () => {
  it('announces loading as a polite status', () => {
    render(<LoadingState />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading...');
  });

  it('renders the custom message when provided', () => {
    render(<LoadingState message="Crunching numbers" />);
    expect(screen.getByRole('status')).toHaveTextContent('Crunching numbers');
  });

  it('hides the spinner from assistive technology', () => {
    const { container } = render(<LoadingState />);
    const spinner = container.querySelector('svg');
    expect(spinner).not.toBeNull();
    expect(spinner).toHaveAttribute('aria-hidden', 'true');
  });
});
