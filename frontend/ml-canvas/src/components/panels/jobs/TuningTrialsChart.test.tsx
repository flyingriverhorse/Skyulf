import React, { isValidElement, cloneElement } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

import { TuningTrialsChart } from './TuningTrialsChart';
import type { TrialPoint } from '../../../core/hooks/useTuningTrials';

// ResponsiveContainer needs a real layout engine (ResizeObserver) that
// jsdom lacks — passthrough with explicit dimensions, same convention as
// FeatureImportanceView.test.tsx.
vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactElement }) => (
      <div style={{ width: 800, height: 240 }}>
        {isValidElement(children)
          ? cloneElement(children, { width: 800, height: 240 } as never)
          : children}
      </div>
    ),
  };
});

const points: TrialPoint[] = [
  { trial: 1, score: 0.5, best: 0.5 },
  { trial: 2, score: 0.9, best: 0.9 },
  { trial: 3, score: 0.7, best: 0.9 },
];

describe('TuningTrialsChart', () => {
  it('renders nothing with fewer than two points', () => {
    const { container } = render(<TuningTrialsChart points={points.slice(0, 1)} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders both series and the metric label', () => {
    render(<TuningTrialsChart points={points} metric="accuracy" />);
    expect(screen.getByText(/Trial score/i)).toBeTruthy();
    expect(screen.getByText(/Best so far/i)).toBeTruthy();
    expect(screen.getByText(/accuracy/i)).toBeTruthy();
  });

  it('shows the live indicator only while live', () => {
    const { rerender } = render(<TuningTrialsChart points={points} isLive />);
    expect(screen.getByText(/live/i)).toBeTruthy();
    rerender(<TuningTrialsChart points={points} />);
    expect(screen.queryByText(/live/i)).toBeNull();
  });

  it('labels iteration series instead of trials when kind is iteration', () => {
    render(<TuningTrialsChart points={points} metric="logloss" kind="iteration" />);
    expect(screen.getByText(/Boosting Iterations/i)).toBeTruthy();
    expect(screen.getByText(/Iteration score/i)).toBeTruthy();
    expect(screen.queryByText(/Tuning Trials/i)).toBeNull();
  });
});
