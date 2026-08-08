import { describe, it, expect } from 'vitest';
import { describeAutosaveUnavailable, RECOVERY_KIND_LABEL } from './canvasRecovery';
import type { CanvasSnapshotDiagnostic } from './canvasPersistence';

describe('describeAutosaveUnavailable (CAN-003)', () => {
  it('returns null when a snapshot is available', () => {
    const diagnostic: CanvasSnapshotDiagnostic = {
      status: 'available',
      snapshot: { version: 1, savedAt: new Date().toISOString(), nodes: [], edges: [] },
    };
    expect(describeAutosaveUnavailable(diagnostic)).toBeNull();
  });

  it('returns null when nothing was ever saved', () => {
    expect(describeAutosaveUnavailable({ status: 'empty' })).toBeNull();
  });

  it('explains a corrupt snapshot without exposing implementation details', () => {
    const result = describeAutosaveUnavailable({ status: 'corrupt' });
    expect(result?.status).toBe('corrupt');
    expect(result?.message).toMatch(/corrupted/i);
    expect(result?.message).not.toMatch(/localStorage|JSON|skyulf:canvas/i);
  });

  it('explains a version-mismatched snapshot', () => {
    const result = describeAutosaveUnavailable({ status: 'version-mismatch', foundVersion: 999 });
    expect(result?.status).toBe('version-mismatch');
    expect(result?.message).toMatch(/incompatible/i);
  });

  it('explains a storage error (quota/disabled) without a stack trace', () => {
    const result = describeAutosaveUnavailable({ status: 'storage-error' });
    expect(result?.status).toBe('storage-error');
    expect(result?.message).toMatch(/full or disabled/i);
  });

  it('labels every recovery source kind', () => {
    expect(RECOVERY_KIND_LABEL.autosave).toBe('Autosave');
    expect(RECOVERY_KIND_LABEL['local-recent']).toBe('Local recent');
    expect(RECOVERY_KIND_LABEL['server-version']).toBe('Server version');
  });
});
