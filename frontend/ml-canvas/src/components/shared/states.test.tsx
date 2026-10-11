import { afterEach, describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
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

  it('disables retry while an async retry is pending', async () => {
    let resolveRetry: (() => void) | undefined;
    const onRetry = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveRetry = () => resolve();
        }),
    );

    render(<ErrorState error="boom" onRetry={onRetry} />);
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));

    expect(screen.getByRole('button', { name: /retry/i })).toBeDisabled();

    if (resolveRetry) {
      resolveRetry();
    }

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /retry/i })).not.toBeDisabled();
    });
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
  afterEach(() => vi.restoreAllMocks());

  it('announces loading as a polite status', () => {
    render(<LoadingState />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading...');
  });

  it('renders the custom message when provided', () => {
    render(<LoadingState message="Crunching numbers" />);
    expect(screen.getByRole('status')).toHaveTextContent('Crunching numbers');
  });

  it.each([false, true])('hides the decorative loading logo with reduced motion %s', reducedMotion => {
    // Both the animated scene and static fallback defer announcements to the status text.
    vi.spyOn(window, 'matchMedia').mockReturnValue(
      Object.assign(new EventTarget(), { matches: reducedMotion }) as MediaQueryList,
    );
    const { container } = render(<LoadingState />);
    const logo = container.querySelector(reducedMotion ? 'img' : 'svg');
    expect(logo).not.toBeNull();
    expect(logo).toHaveAttribute('aria-hidden', 'true');
    expect(screen.queryByRole('img')).toBeNull();
  });
});
