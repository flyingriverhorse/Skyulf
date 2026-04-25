import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useState } from 'react';
import { ErrorBoundary } from './ErrorBoundary';

const Boom: React.FC<{ throwNow: boolean; msg?: string }> = ({ throwNow, msg = 'kapow' }) => {
  if (throwNow) throw new Error(msg);
  return <div>safe-child</div>;
};

describe('ErrorBoundary', () => {
  // React logs caught errors to console.error in dev; silence to keep
  // test output clean.
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders children normally when nothing throws', () => {
    render(
      <ErrorBoundary>
        <Boom throwNow={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByText('safe-child')).toBeTruthy();
  });

  it('renders the default fallback and surfaces the error message when a child throws', () => {
    render(
      <ErrorBoundary>
        <Boom throwNow msg="boom-detail" />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/something went wrong/i)).toBeTruthy();
    expect(screen.getByText(/boom-detail/)).toBeTruthy();
  });

  it('invokes onError with the caught error', () => {
    const onError = vi.fn();
    render(
      <ErrorBoundary onError={onError}>
        <Boom throwNow msg="hook-me" />
      </ErrorBoundary>,
    );
    expect(onError).toHaveBeenCalledTimes(1);
    expect((onError.mock.calls[0]?.[0] as Error).message).toBe('hook-me');
  });

  it('"Try again" resets the boundary so a now-safe child renders', () => {
    const Harness: React.FC = () => {
      const [throwNow, setThrowNow] = useState(true);
      return (
        <>
          <button onClick={() => setThrowNow(false)}>fix-it</button>
          <ErrorBoundary>
            <Boom throwNow={throwNow} />
          </ErrorBoundary>
        </>
      );
    };
    render(<Harness />);
    expect(screen.getByText(/something went wrong/i)).toBeTruthy();

    fireEvent.click(screen.getByText('fix-it'));
    fireEvent.click(screen.getByRole('button', { name: /try again/i }));

    expect(screen.getByText('safe-child')).toBeTruthy();
  });

  it('renders a custom fallback when provided', () => {
    render(
      <ErrorBoundary fallback={(err) => <div>custom:{err.message}</div>}>
        <Boom throwNow msg="x" />
      </ErrorBoundary>,
    );
    expect(screen.getByText('custom:x')).toBeTruthy();
  });
});
