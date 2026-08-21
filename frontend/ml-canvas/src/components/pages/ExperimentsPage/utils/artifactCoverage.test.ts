import { describe, expect, it } from 'vitest';

import { getArtifactCoverage } from './artifactCoverage';

describe('getArtifactCoverage', () => {
  it('marks segmentation as unsupported for a classification run', () => {
    const result = getArtifactCoverage('segmentation', {
      task: 'classification',
      status: 'completed',
      hasArtifact: false,
    });
    expect(result.status).toBe('unsupported');
    expect(result.reason).toMatch(/not a segmentation/i);
  });

  it('marks feature importance as unsupported for a clustering run', () => {
    const result = getArtifactCoverage('feature_importance', {
      task: 'segmentation',
      status: 'completed',
      hasArtifact: false,
    });
    expect(result.status).toBe('unsupported');
  });

  it('marks a supported run as failed when status is "failed"', () => {
    const result = getArtifactCoverage('shap', {
      task: 'classification',
      status: 'failed',
      hasArtifact: false,
    });
    expect(result.status).toBe('failed');
  });

  it('marks a run as failed when it has an error message even without a "failed" status', () => {
    const result = getArtifactCoverage('shap', {
      task: 'classification',
      status: 'completed',
      error: 'trainer crashed mid-explain',
      hasArtifact: false,
    });
    expect(result.status).toBe('failed');
    expect(result.reason).toContain('trainer crashed mid-explain');
  });

  it('marks a still-running supported run as not_computed', () => {
    const result = getArtifactCoverage('feature_importance', {
      task: 'classification',
      status: 'running',
      hasArtifact: false,
    });
    expect(result.status).toBe('not_computed');
    expect(result.reason).toMatch(/has not finished/i);
  });

  it('marks a completed supported run missing the artifact as not_computed', () => {
    const result = getArtifactCoverage('feature_importance', {
      task: 'classification',
      status: 'completed',
      hasArtifact: false,
    });
    expect(result.status).toBe('not_computed');
    expect(result.reason).toMatch(/not supported for this model type/i);
  });

  it('marks a completed supported run with the artifact as available', () => {
    const result = getArtifactCoverage('segmentation', {
      task: 'segmentation',
      status: 'completed',
      hasArtifact: true,
    });
    expect(result.status).toBe('available');
  });

  it('treats "succeeded" the same as "completed" as a terminal success status', () => {
    const result = getArtifactCoverage('shap', {
      task: 'regression',
      status: 'succeeded',
      hasArtifact: true,
    });
    expect(result.status).toBe('available');
  });
});
