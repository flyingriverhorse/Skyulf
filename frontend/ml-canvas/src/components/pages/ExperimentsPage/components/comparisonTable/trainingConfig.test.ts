import { expect, it } from 'vitest';
import type { JobInfo } from '../../../../../core/api/jobs';
import { trainingConfigValue } from './trainingConfig';

const job: JobInfo = {
  job_id: 'job-1', pipeline_id: 'pipeline-1', node_id: 'model-1',
  job_type: 'training', status: 'completed', start_time: null, end_time: null,
  error: null, result: null, created_at: '2026-09-10', model_type: 'random_forest',
};

it.each(['unknown', '__proto__', 'constructor', 'toString', 'hasOwnProperty'])(
  'uses the missing-field fallback for an unsupported field %s', field => {
    // Field lookup must not treat inherited Object members as value formatters.
    expect(trainingConfigValue(field, job, { cv_enabled: true, tuning_config: { metric: 'accuracy' } })).toBe('-');
  },
);
