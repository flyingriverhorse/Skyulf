import React from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

export interface ErrorBoundaryProps {
  children: React.ReactNode;
  /** Optional render-prop fallback. If omitted, the built-in card is used. */
  fallback?: (error: Error, reset: () => void) => React.ReactNode;
  /** Called once on each caught error. Useful for logging to Sentry/etc. */
  onError?: (error: Error, info: React.ErrorInfo) => void;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/**
 * Single global error boundary used at the app root and around every
 * `<Suspense>` route. Without this, a single render throw — e.g. a
 * `definition.validate(data)` crash — render-blanks the whole canvas
 * with no recovery. The fallback offers a "Try again" button that
 * resets the boundary so the user doesn't have to reload the tab.
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // Always surface to the console — production builds strip our debug
    // logs but an uncaught render error is worth keeping.
    console.error('[ErrorBoundary]', error, info);
    this.props.onError?.(error, info);
  }

  reset = (): void => {
    this.setState({ error: null });
  };

  render(): React.ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    if (this.props.fallback) return this.props.fallback(error, this.reset);

    return (
      <div
        role="alert"
        className="flex h-full w-full items-center justify-center p-8"
      >
        <div className="max-w-lg rounded-xl border border-red-200 bg-red-50 p-6 dark:border-red-900/50 dark:bg-red-950/40">
          <div className="mb-3 flex items-center gap-3">
            <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
            <h2 className="text-lg font-semibold text-red-900 dark:text-red-100">
              Something went wrong
            </h2>
          </div>
          <p className="mb-4 text-sm text-red-800 dark:text-red-200">
            {error.message || 'An unexpected error occurred while rendering this view.'}
          </p>
          <button
            type="button"
            onClick={this.reset}
            className="inline-flex items-center gap-2 rounded-md bg-red-600 px-3 py-2 text-sm font-medium text-white hover:bg-red-700 focus-ring"
          >
            <RotateCcw className="h-4 w-4" />
            Try again
          </button>
        </div>
      </div>
    );
  }
}
