import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ClusteringTab } from './ClusteringTab';

describe('ClusteringTab', () => {
  it.each([{ points: [] }, { points: [{ x: 1, y: 2, cluster: 0 }] }])('describes and downloads the displayed summary with points $points', ({ points }) => {
    // Stored projection points must not imply a scatter plot exists in this summary view.
    const download = vi.fn();
    render(<ClusteringTab profile={{ clustering: {
      method: 'KMeans', n_clusters: 1, inertia: null, points,
      clusters: [{ cluster_id: 0, size: 4, percentage: 100, center: { amount: null, other: 3 } }],
    } }} downloadChart={download} />);
    expect(screen.queryByText(/plot shows a 2D projection/i)).not.toBeInTheDocument();
    expect(screen.getByText(/summary shows the clustering method/i)).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Download Summary' }));
    expect(download).toHaveBeenCalledWith('clustering-chart', 'clustering-summary', 'Clustering Summary', '1 Clusters Found');
    expect(screen.getAllByText('N/A')).toHaveLength(2);
  });
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
