import { expect, it } from 'vitest';
import { buildBaseTuningConfig, buildFixedTrainingParams } from './training';

/** Both model routes must retain the separate inner-fold choice without adding defaults. */
it.each([buildBaseTuningConfig, buildFixedTrainingParams])('forwards explicit nested folds', (build) => {
  expect(build({ cv_type: 'nested_cv', cv_folds: 5, cv_inner_folds: 3 })).toMatchObject({ cv_type: 'nested_cv', cv_folds: 5, cv_inner_folds: 3 });
  expect(build({ cv_type: 'k_fold', cv_folds: 5 })).not.toHaveProperty('cv_inner_folds');
});

/** Policy and metadata must survive both fixed and searched payloads. */
it.each([buildBaseTuningConfig, buildFixedTrainingParams])('forwards nested split policies', (build) => {
  const policy = { cv_nested_type: 'time_series_split', cv_group_column: 'customer', cv_gap: 2, cv_test_size: 8, cv_max_train_size: 30 };
  expect(build({ cv_type: 'nested_cv', ...policy })).toMatchObject(policy);
});

/** Fixed training must send a JSON null class weight to the API after choosing None. */
it('preserves a null class weight in the fixed training payload', () => {
  const payload = buildFixedTrainingParams({ hyperparameters: { class_weight: null } });
  expect(JSON.parse(JSON.stringify(payload))).toMatchObject({ hyperparameters: { class_weight: null } });
});
