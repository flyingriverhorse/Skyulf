import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import React, { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';

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

const baseProfile: EDAProfile = {
  row_count: 2,
  column_count: 3,
  columns: {
    x: { name: 'x', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 },
    y: { name: 'y', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 },
    group: { name: 'group', dtype: 'Categorical', missing_count: 0, missing_percentage: 0 },
  },
  sample_data: [
    { x: 1, y: 2, group: 'a' },
    { x: 3, y: 4, group: 'b' },
  ],
};

const pcaProfile: EDAProfile = {
  row_count: 2,
  column_count: 2,
  columns: baseProfile.columns,
  pca_data: [
    { x: 1, y: 2, z: 3, label: 'a' },
    { x: 4, y: 5, z: 6, label: 'b' },
  ],
};

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function BivariateHarness({ profile }: { profile: EDAProfile }) {
  const [scatterX, setScatterX] = useState('');
  const [scatterY, setScatterY] = useState('');
  const [scatterZ, setScatterZ] = useState('');
  const [scatterColor, setScatterColor] = useState('');
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

describe('EDA chart download buttons', () => {
  it('keeps the bivariate download button disabled until X and Y are selected', () => {
    renderWithClient(<BivariateHarness profile={baseProfile} />);

    const button = screen.getByRole('button', { name: /select x and y variables to enable download/i });
    expect(button).toBeDisabled();
  });

  it('enables the bivariate download button after X and Y are selected', () => {
    renderWithClient(<BivariateHarness profile={baseProfile} />);

    const selects = screen.getAllByRole('combobox');
    const xSelect = selects[0]!;
    const ySelect = selects[1]!;

    fireEvent.change(xSelect, { target: { value: 'x' } });
    fireEvent.change(ySelect, { target: { value: 'y' } });

    expect(screen.getByRole('button', { name: /download chart/i })).toBeEnabled();
  });

  it('keeps the PCA download button disabled until PCA data is available', () => {
    renderWithClient(<PCAHarness profile={baseProfile} />);

    const button = screen.getByRole('button', { name: /not enough numeric data for pca/i });
    expect(button).toBeDisabled();
  });

  it('enables the PCA download button when PCA data is present', () => {
    renderWithClient(<PCAHarness profile={pcaProfile} />);

    expect(screen.getByRole('button', { name: /download chart/i })).toBeEnabled();
  });
});
