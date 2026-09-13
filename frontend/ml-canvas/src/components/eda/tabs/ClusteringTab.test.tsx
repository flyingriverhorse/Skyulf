import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ClusteringTab } from './ClusteringTab';

describe('ClusteringTab', () => {
  it('renders unavailable inertia and centroid values without turning them into zero', () => {
    /** The finite JSON serializer can replace overflowing numeric analysis results with null. */
    render(<ClusteringTab profile={{ clustering: {
      method: 'KMeans', n_clusters: 1, inertia: null, points: [],
      clusters: [{ cluster_id: 0, size: 4, percentage: 100, center: { amount: null, other: 3 } }],
    } }} downloadChart={vi.fn()} />);
    expect(screen.getAllByText('N/A')).toHaveLength(2);
    expect(screen.getByText('3.00')).toBeInTheDocument();
  });
});
