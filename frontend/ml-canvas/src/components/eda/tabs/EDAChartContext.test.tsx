import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, within } from '@testing-library/react';
import React, { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EDAProfile } from '../../../core/types/edaProfile';
import { BivariateTab } from './BivariateTab';
import { PCATab } from './PCATab';

vi.mock('../../ui/InfoTooltip', () => ({
  InfoTooltip: () => <span data-testid="info-tooltip" />,
}));

vi.mock('../CanvasScatterPlot', () => ({
  CanvasScatterPlot: () => <div data-testid="canvas-scatter-plot" />,
}));

vi.mock('../ThreeDScatterPlot', () => ({
  ThreeDScatterPlot: () => <div data-testid="three-d-scatter-plot" />,
}));

const groupedProfile: EDAProfile = {
  row_count: 4,
  column_count: 3,
  columns: {
    x: { name: 'x', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 },
    y: { name: 'y', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 },
    species: { name: 'species', dtype: 'Categorical', missing_count: 0, missing_percentage: 0 },
  },
  sample_data: [
    { x: 1, y: 2, species: 'setosa' },
    { x: 3, y: 4, species: 'versicolor' },
    { x: 5, y: 1, species: 'virginica' },
    { x: 2, y: 6, species: 'setosa' },
  ],
};

const pcaProfile: EDAProfile = {
  row_count: 2,
  column_count: 2,
  columns: groupedProfile.columns,
  pca_data: [
    { x: 1, y: 2, z: 3, label: 'cluster-a' },
    { x: 4, y: 5, z: 6, label: 'cluster-b' },
  ],
};

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function BivariateHarness({ profile }: { profile: EDAProfile }) {
  const [scatterX, setScatterX] = useState('x');
  const [scatterY, setScatterY] = useState('y');
  const [scatterZ, setScatterZ] = useState('');
  const [scatterColor, setScatterColor] = useState('species');
  const [is3D, setIs3D] = useState(false);

  return (
    <BivariateTab
      profile={profile}
      downloadChart={vi.fn()}
      scatterX={scatterX}
      setScatterX={setScatterX}
      scatterY={scatterY}
      setScatterY={setScatterY}
      scatterZ={scatterZ}
      setScatterZ={setScatterZ}
      scatterColor={scatterColor}
      setScatterColor={setScatterColor}
      is3D={is3D}
      setIs3D={setIs3D}
    />
  );
}

function PCAHarness({ profile }: { profile: EDAProfile }) {
  const [isPCA3D, setIsPCA3D] = useState(false);
  return <PCATab profile={profile} isPCA3D={isPCA3D} setIsPCA3D={setIsPCA3D} downloadChart={vi.fn()} />;
}

afterEach(() => {
  document.documentElement.classList.remove('dark');
});

describe('EDA scatter chart context (legend + table alternative)', () => {
  it('renders a color+shape legend for the bivariate scatter groups', () => {
    renderWithClient(<BivariateHarness profile={groupedProfile} />);
    // 3 distinct species groups -> 3 legend entries with shape swatches.
    const svgs = document.querySelectorAll('svg[aria-hidden="true"]');
    expect(svgs.length).toBeGreaterThanOrEqual(3);
    expect(screen.getByText('setosa')).toBeInTheDocument();
    expect(screen.getByText('versicolor')).toBeInTheDocument();
  });

  it('provides a data table alternative for the bivariate chart with the selected variables', () => {
    renderWithClient(<BivariateHarness profile={groupedProfile} />);
    fireEvent.click(screen.getByRole('button', { name: /view data table/i }));
    const region = screen.getByRole('region', { name: /bivariate scatter plot data/i });
    expect(within(region).getByRole('columnheader', { name: 'x' })).toBeInTheDocument();
    expect(within(region).getByRole('columnheader', { name: 'species' })).toBeInTheDocument();
  });

  it('renders a data table alternative for the PCA projection', () => {
    renderWithClient(<PCAHarness profile={pcaProfile} />);
    fireEvent.click(screen.getByRole('button', { name: /view data table/i }));
    const region = screen.getByRole('region', { name: /pca projection data/i });
    expect(within(region).getByText('cluster-a')).toBeInTheDocument();
  });

  it('keeps legend and table controls present at a narrow (390px) viewport', () => {
    Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 390 });
    window.dispatchEvent(new Event('resize'));

    renderWithClient(<BivariateHarness profile={groupedProfile} />);
    expect(screen.getByRole('button', { name: /view data table/i })).toBeInTheDocument();
    expect(screen.getByText('setosa')).toBeInTheDocument();
  });

  it('keeps legend and table controls present in dark mode', () => {
    document.documentElement.classList.add('dark');
    renderWithClient(<BivariateHarness profile={groupedProfile} />);
    expect(screen.getByRole('button', { name: /view data table/i })).toBeInTheDocument();
    expect(screen.getByText('setosa')).toBeInTheDocument();
  });
});
