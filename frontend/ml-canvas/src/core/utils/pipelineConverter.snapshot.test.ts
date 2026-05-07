/**
 * Snapshot coverage for `convertGraphToPipelineConfig`.
 *
 * Why a separate file: granular `toEqual` / `toMatchObject` tests are good
 * at pinning behaviour we *intend*, but they miss accidental changes to
 * fields nobody asserted on (extra params, key ordering inside `metadata`,
 * silently-renamed `step_type`, …). Snapshots make every refactor that
 * changes the canvas → backend payload show up as a diff in PR review,
 * and an intentional change is a one-liner: `npm run test -- -u`.
 *
 * The non-deterministic `pipeline_id` (UUID per call) is normalised before
 * snapshotting so reruns are stable. Inline snapshots keep the assertion
 * and the input side-by-side — no `__snapshots__/` directory to chase.
 */
import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { convertGraphToPipelineConfig } from './pipelineConverter';

const node = (id: string, definitionType: string, extra: Record<string, unknown> = {}): Node => ({
    id,
    type: 'custom',
    position: { x: 0, y: 0 },
    data: { definitionType, ...extra },
});

const edge = (source: string, target: string): Edge => ({
    id: `${source}-${target}`,
    source,
    target,
});

/** Replace UUID-suffixed pipeline_id so snapshots are stable across runs. */
const normaliseConfig = (cfg: ReturnType<typeof convertGraphToPipelineConfig>) => ({
    ...cfg,
    pipeline_id: 'preview_<uuid>',
});

describe('convertGraphToPipelineConfig — snapshots', () => {
    it('canonical training pipeline: dataset → imputer → split → train', () => {
        const nodes = [
            node('ds', 'dataset_node', { datasetId: 'd1' }),
            node('imp', 'imputation_node', {
                method: 'simple',
                strategy: 'mean',
                columns: ['x', 'y'],
            }),
            node('train', 'basic_training', {
                target_column: 'target',
                model_type: 'random_forest_classifier',
                hyperparameters: { n_estimators: 100 },
                cv_enabled: true,
                cv_folds: 5,
                cv_type: 'kfold',
                cv_shuffle: true,
                cv_random_state: 42,
            }),
        ];
        const edges = [edge('ds', 'imp'), edge('imp', 'train')];

        expect(normaliseConfig(convertGraphToPipelineConfig(nodes, edges))).toMatchSnapshot();
    });

    it('parallel preview branches with different depths are all preserved', () => {
        const nodes = [
            node('ds', 'dataset_node', { datasetId: 'd1' }),
            node('impA', 'imputation_node', { method: 'simple', strategy: 'mean' }),
            node('impB', 'imputation_node', { method: 'simple', strategy: 'median' }),
            node('scaleB', 'StandardScaler', { columns: ['x'] }),
        ];
        const edges = [edge('ds', 'impA'), edge('ds', 'impB'), edge('impB', 'scaleB')];

        expect(normaliseConfig(convertGraphToPipelineConfig(nodes, edges))).toMatchSnapshot();
    });

    it('KNN imputer routes to KNNImputer step_type with hyperparams', () => {
        const nodes = [
            node('ds', 'dataset_node', { datasetId: 'd1' }),
            node('knn', 'imputation_node', {
                method: 'knn',
                columns: ['x', 'y'],
                n_neighbors: 7,
                weights: 'distance',
            }),
        ];
        const edges = [edge('ds', 'knn')];

        expect(normaliseConfig(convertGraphToPipelineConfig(nodes, edges))).toMatchSnapshot();
    });

    it('mixed training + dangling preview branch keeps both leaves', () => {
        const nodes = [
            node('ds', 'dataset_node', { datasetId: 'd1' }),
            node('impA', 'imputation_node', { method: 'simple', strategy: 'mean' }),
            node('train', 'basic_training', {
                target_column: 'y',
                model_type: 'rf',
                hyperparameters: {},
                cv_enabled: false,
                cv_folds: 5,
                cv_type: 'kfold',
                cv_shuffle: true,
                cv_random_state: 0,
            }),
            node('scaleB', 'StandardScaler', { columns: ['x'] }),
        ];
        const edges = [edge('ds', 'impA'), edge('impA', 'train'), edge('ds', 'scaleB')];

        expect(normaliseConfig(convertGraphToPipelineConfig(nodes, edges))).toMatchSnapshot();
    });

    it('non-default merge strategy survives as underscore-prefixed param', () => {
        const nodes = [
            node('ds', 'dataset_node', { datasetId: 'd1' }),
            node('imp', 'imputation_node', {
                method: 'simple',
                strategy: 'mean',
                merge_strategy: 'first_wins',
            }),
        ];
        const edges = [edge('ds', 'imp')];

        expect(normaliseConfig(convertGraphToPipelineConfig(nodes, edges))).toMatchSnapshot();
    });
});
