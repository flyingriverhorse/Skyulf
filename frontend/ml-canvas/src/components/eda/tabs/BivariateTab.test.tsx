import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { EDAProfile } from '../../../core/types/edaProfile';
import { BivariateTab } from './BivariateTab';
import { PCATab } from './PCATab';

const charts = vi.hoisted(() => ({ two: vi.fn(), three: vi.fn() }));
vi.mock('../CanvasScatterPlot', () => ({ CanvasScatterPlot: (props: unknown) => { charts.two(props); return <div />; } }));
vi.mock('../ThreeDScatterPlot', () => ({ ThreeDScatterPlot: (props: unknown) => { charts.three(props); return <div />; } }));

/** Real tabs keep their legend, table and control behavior around the chart boundary. */
function wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>;
}

describe('Statistical chart series and selection', () => {
    it('keeps bivariate coordinates, labels and missing-Z fallback across selections', () => {
        /** Missing Z still produces 2D, while an explicit Z preserves original sample orientation. */
        const profile: EDAProfile = { row_count: 2, column_count: 3, columns: {
            a: { name: 'a', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 },
            b: { name: 'b', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 },
            group: { name: 'group', dtype: 'Categorical', missing_count: 0, missing_percentage: 0 },
        }, sample_data: [{ a: 0, b: 4, group: 'one' }, { a: -2, b: 8, group: 'two' }] };
        const props = { profile, downloadChart: vi.fn(), scatterX: 'b', setScatterX: vi.fn(), scatterY: 'a', setScatterY: vi.fn(), scatterZ: '', setScatterZ: vi.fn(), scatterColor: 'group', setScatterColor: vi.fn(), is3D: true, setIs3D: vi.fn() };
        const { rerender } = render(<BivariateTab {...props} />, { wrapper });
        expect(charts.two).toHaveBeenLastCalledWith(expect.objectContaining({ data: profile.sample_data, xKey: 'b', yKey: 'a', xLabel: 'b', yLabel: 'a', labelKey: 'group' }));
        fireEvent.change(screen.getAllByRole('combobox')[0]!, { target: { value: 'a' } });
        expect(props.setScatterX).toHaveBeenCalledWith('a');
        rerender(<BivariateTab {...props} scatterZ="b" />);
        expect(charts.three).toHaveBeenLastCalledWith(expect.objectContaining({ data: profile.sample_data, xKey: 'b', yKey: 'a', zKey: 'b', zLabel: 'b' }));
        fireEvent.click(screen.getByRole('button', { name: 'Download Chart' }));
        expect(props.downloadChart).toHaveBeenCalledWith('bivariate-chart', 'bivariate-analysis', 'Bivariate Analysis', 'b vs a');
        rerender(<BivariateTab {...props} scatterX="" />);
        expect(screen.getByText('Select X and Y variables to generate scatter plot.')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /enable download/ })).toBeDisabled();
    });

    it('preserves PCA projections, zero loadings and empty versus absent data', () => {
        /** PCA coordinate names and component slicing must stay aligned with the selected dimension. */
        const profile = { pca_data: [{ x: 0, y: -2, z: 3, label: 'one' }], pca_components: [
            { component: 'PC1', explained_variance_ratio: 0.5, top_features: { zero: 0, positive: 0.2 } },
            { component: 'PC2', explained_variance_ratio: 0.3 },
            { component: 'PC3', explained_variance_ratio: 0.2 },
        ] };
        const props = { profile, isPCA3D: false, setIsPCA3D: vi.fn(), downloadChart: vi.fn() };
        const { rerender } = render(<PCATab {...props} />, { wrapper });
        expect(charts.two).toHaveBeenLastCalledWith(expect.objectContaining({ data: profile.pca_data, xKey: 'x', yKey: 'y', xLabel: 'Principal Component 1 (PC1)', yLabel: 'Principal Component 2 (PC2)' }));
        expect(screen.getByText('0.000')).toHaveClass('text-red-500');
        expect(screen.queryByText('PC3')).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Switch to 3D' }));
        expect(props.setIsPCA3D).toHaveBeenCalledWith(true);
        rerender(<PCATab {...props} isPCA3D />);
        expect(charts.three).toHaveBeenLastCalledWith(expect.objectContaining({ data: profile.pca_data, xKey: 'x', yKey: 'y', zKey: 'z', xLabel: 'PC1', yLabel: 'PC2', zLabel: 'PC3' }));
        expect(screen.getByText('PC3')).toBeInTheDocument();
        rerender(<PCATab {...props} profile={{ pca_data: [] }} />);
        expect(charts.two).toHaveBeenLastCalledWith(expect.objectContaining({ data: [] }));
        expect(screen.queryByText('Not enough numeric data for PCA.')).not.toBeInTheDocument();
        rerender(<PCATab {...props} profile={{}} />);
        expect(screen.getByText('Not enough numeric data for PCA.')).toBeInTheDocument();
    });
});
