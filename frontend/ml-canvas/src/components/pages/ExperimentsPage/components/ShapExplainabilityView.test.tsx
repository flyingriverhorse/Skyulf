import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ShapExplainabilityView, type ShapExplanationEntry } from './ShapExplainabilityView';

const scatter = vi.hoisted(() => vi.fn());
vi.mock('recharts', async () => {
    const actual = await vi.importActual<typeof import('recharts')>('recharts');
    return { ...actual,
        ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
        ScatterChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
        Scatter: (props: unknown) => { scatter(props); return null; },
    };
});

/** Distinct run artifacts expose accidental run, feature or matrix transposition. */
function entries(): ShapExplanationEntry[] {
    return [
        { jobId: 'a', pipeline_id: 'preview_aaaaaaaa', modelType: 'forest', shapExplanation: {
            feature_names: ['first', 'second'], mean_abs_importance: { first: 4, second: 2 },
            samples: [{ base_value: 0, feature_values: { first: 10 }, shap_values: { first: -4, second: 2 } }],
            interactions: { feature_names: ['first', 'second'], matrix: [[1, 2], [3, 4]] },
        } },
        { jobId: 'b', pipeline_id: 'preview_bbbbbbbb', modelType: 'unknown', shapExplanation: {
            feature_names: ['other'], mean_abs_importance: { other: 7 },
            samples: [{ base_value: 1, feature_values: { other: 8 }, shap_values: { other: 7 } }],
        } },
        { jobId: 'missing', pipeline_id: 'preview_cccccccc', modelType: 'svm', shapExplanation: null },
    ];
}

describe('ShapExplainabilityView public run routing', () => {
    it('retains run and feature selections, missing-value fallbacks and interaction orientation', () => {
        /** Actual SHAP consumers must receive the selected artifact without losing their series semantics. */
        const props = { shapExplanationByJob: entries(), coverageInputs: [], handleDownload: vi.fn(), downloadingChart: null, doneChart: null };
        const { rerender } = render(<ShapExplainabilityView {...props} />);
        fireEvent.click(screen.getByRole('button', { name: 'Dependence' }));
        expect(scatter).toHaveBeenLastCalledWith(expect.objectContaining({ data: [{ featureValue: 10, shapValue: -4 }] }));
        fireEvent.change(screen.getAllByRole('combobox')[1]!, { target: { value: 'second' } });
        expect(scatter).toHaveBeenLastCalledWith(expect.objectContaining({ data: [{ featureValue: 0, shapValue: 2 }] }));
        fireEvent.change(screen.getAllByRole('combobox')[0]!, { target: { value: 'b' } });
        expect(scatter).toHaveBeenLastCalledWith(expect.objectContaining({ data: [{ featureValue: 8, shapValue: 7 }] }));
        fireEvent.click(screen.getByRole('button', { name: 'Interaction' }));
        expect(screen.getByText(/Interaction values are not available/)).toBeInTheDocument();
        fireEvent.change(screen.getByRole('combobox'), { target: { value: 'a' } });
        expect(screen.getByTitle('first × second: 2.0000')).toHaveTextContent('2.00');
        expect(screen.getByTitle('second × first: 3.0000')).toHaveTextContent('3.00');
        fireEvent.click(screen.getByRole('button', { name: 'Beeswarm' }));
        expect(scatter).toHaveBeenLastCalledWith(expect.objectContaining({ data: [
            expect.objectContaining({ feature: 'first', shapValue: -4, featureValue: 10 }),
            expect.objectContaining({ feature: 'second', shapValue: 2, featureValue: 0 }),
        ] }));
        rerender(<ShapExplainabilityView {...props} shapExplanationByJob={[props.shapExplanationByJob[1]!]} />);
        expect(scatter).toHaveBeenLastCalledWith(expect.objectContaining({ data: [expect.objectContaining({ feature: 'other', shapValue: 7 })] }));
        rerender(<ShapExplainabilityView {...props} shapExplanationByJob={[props.shapExplanationByJob[2]!]} />);
        expect(screen.queryByRole('button', { name: 'Summary' })).not.toBeInTheDocument();
    });
});
