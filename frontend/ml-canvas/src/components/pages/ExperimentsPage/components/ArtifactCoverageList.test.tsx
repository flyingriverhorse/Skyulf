import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { ArtifactCoverageList, type ArtifactCoverageEntry } from './ArtifactCoverageList';

afterEach(() => {
  document.documentElement.classList.remove('dark');
});

describe('ArtifactCoverageList', () => {
  it('renders nothing for an empty entry list', () => {
    const { container } = render(<ArtifactCoverageList entries={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders icon + text for every status so availability is never color-only', () => {
    const entries: ArtifactCoverageEntry[] = [
      { jobId: 'a', label: 'random_forest (a1b2c3d4)', status: 'available', reason: 'Feature importance is available for this run.' },
      { jobId: 'b', label: 'kmeans (b2c3d4e5)', status: 'not_computed', reason: 'Feature importance was not computed for this run.' },
      { jobId: 'c', label: 'kmeans (c3d4e5f6)', status: 'unsupported', reason: 'Feature importance does not apply to this run.' },
      { jobId: 'd', label: 'xgboost (d4e5f6a7)', status: 'failed', reason: 'Run failed before this artifact could be produced.' },
    ];
    render(<ArtifactCoverageList entries={entries} />);

    expect(screen.getByText('Available')).toBeInTheDocument();
    expect(screen.getByText('Not computed')).toBeInTheDocument();
    expect(screen.getByText('Unsupported')).toBeInTheDocument();
    expect(screen.getByText('Failed')).toBeInTheDocument();

    entries.forEach((entry) => {
      expect(screen.getByText(entry.label)).toBeInTheDocument();
      expect(screen.getByText(entry.reason)).toBeInTheDocument();
    });

    // Each status row carries an svg icon alongside its text label.
    const rows = screen.getAllByRole('row');
    expect(rows).toHaveLength(4);
    rows.forEach((row) => {
      expect(row.querySelector('svg')).toBeTruthy();
    });
  });

  it('exposes a table/row role structure for assistive tech', () => {
    render(
      <ArtifactCoverageList
        entries={[{ jobId: 'a', label: 'run-a', status: 'available', reason: 'ok' }]}
      />,
    );
    expect(screen.getByRole('table', { name: /artifact availability by run/i })).toBeInTheDocument();
  });

  it('renders correctly in dark mode', () => {
    document.documentElement.classList.add('dark');
    render(
      <ArtifactCoverageList
        entries={[{ jobId: 'a', label: 'run-a', status: 'failed', reason: 'crashed' }]}
      />,
    );
    expect(screen.getByText('Failed')).toBeInTheDocument();
  });

  it('renders correctly at a 390px viewport', () => {
    Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 390 });
    window.dispatchEvent(new Event('resize'));
    render(
      <ArtifactCoverageList
        entries={[{ jobId: 'a', label: 'run-a', status: 'not_computed', reason: 'pending' }]}
      />,
    );
    expect(screen.getByText('Not computed')).toBeInTheDocument();
  });
});
