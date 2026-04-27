import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  getRecentPipelines,
  pushRecentPipeline,
  clearRecentPipelines,
  togglePinRecentPipeline,
  renameRecentPipeline,
  deleteRecentPipeline,
} from './recentPipelines';

const LS_KEY = 'skyulf:canvas:recent:v1';

describe('recentPipelines', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('returns [] when nothing is stored', () => {
    expect(getRecentPipelines()).toEqual([]);
  });

  it('returns [] when payload is corrupt', () => {
    window.localStorage.setItem(LS_KEY, 'not-json');
    expect(getRecentPipelines()).toEqual([]);
  });

  it('pushes entries newest-first', () => {
    pushRecentPipeline({ name: 'A', nodes: [], edges: [] });
    pushRecentPipeline({ name: 'B', nodes: [], edges: [] });
    const list = getRecentPipelines();
    expect(list.map((e) => e.name)).toEqual(['B', 'A']);
  });

  it('dedupes same-name re-saves (newest wins)', () => {
    pushRecentPipeline({ name: 'A', nodes: [], edges: [], savedAt: '2026-01-01T00:00:00Z' });
    pushRecentPipeline({ name: 'B', nodes: [], edges: [] });
    pushRecentPipeline({ name: 'A', nodes: [], edges: [], savedAt: '2026-04-01T00:00:00Z' });
    const list = getRecentPipelines();
    expect(list.map((e) => e.name)).toEqual(['A', 'B']);
    expect(list[0]?.savedAt).toBe('2026-04-01T00:00:00Z');
  });

  it('caps the unpinned buffer at 5 entries', () => {
    for (let i = 0; i < 8; i++) {
      pushRecentPipeline({ name: `pipe-${i}`, nodes: [], edges: [] });
    }
    const list = getRecentPipelines();
    expect(list).toHaveLength(5);
    expect(list[0]?.name).toBe('pipe-7');
    expect(list[4]?.name).toBe('pipe-3');
  });

  it('togglePinRecentPipeline pins and unpins; pin survives FIFO eviction', () => {
    pushRecentPipeline({ name: 'keep-me', nodes: [], edges: [] });
    const id = getRecentPipelines()[0]!.id;
    togglePinRecentPipeline(id);
    for (let i = 0; i < 8; i++) {
      pushRecentPipeline({ name: `pipe-${i}`, nodes: [], edges: [] });
    }
    const list = getRecentPipelines();
    // 1 pinned + 5 unpinned
    expect(list).toHaveLength(6);
    // Pinned floats to the top
    expect(list[0]?.name).toBe('keep-me');
    expect(list[0]?.pinned).toBe(true);
    // Unpinning then pushing more evicts it
    togglePinRecentPipeline(list[0]!.id);
    for (let i = 0; i < 8; i++) {
      pushRecentPipeline({ name: `more-${i}`, nodes: [], edges: [] });
    }
    expect(getRecentPipelines().some((e) => e.name === 'keep-me')).toBe(false);
  });

  it('renameRecentPipeline renames and rejects empty / clashing names', () => {
    pushRecentPipeline({ name: 'A', nodes: [], edges: [] });
    pushRecentPipeline({ name: 'B', nodes: [], edges: [] });
    const aId = getRecentPipelines().find((e) => e.name === 'A')!.id;
    renameRecentPipeline(aId, 'A-renamed');
    expect(getRecentPipelines().find((e) => e.id === aId)?.name).toBe('A-renamed');
    // Empty name is a no-op
    renameRecentPipeline(aId, '   ');
    expect(getRecentPipelines().find((e) => e.id === aId)?.name).toBe('A-renamed');
    // Clash with another entry's name is a no-op
    renameRecentPipeline(aId, 'B');
    expect(getRecentPipelines().find((e) => e.id === aId)?.name).toBe('A-renamed');
  });

  it('deleteRecentPipeline removes a single entry by id', () => {
    pushRecentPipeline({ name: 'A', nodes: [], edges: [] });
    pushRecentPipeline({ name: 'B', nodes: [], edges: [] });
    const bId = getRecentPipelines().find((e) => e.name === 'B')!.id;
    deleteRecentPipeline(bId);
    expect(getRecentPipelines().map((e) => e.name)).toEqual(['A']);
  });

  it('clearRecentPipelines wipes storage', () => {
    pushRecentPipeline({ name: 'A', nodes: [], edges: [] });
    clearRecentPipelines();
    expect(getRecentPipelines()).toEqual([]);
  });

  it('preserves datasetId when provided', () => {
    pushRecentPipeline({ name: 'A', nodes: [], edges: [], datasetId: 'ds-1' });
    expect(getRecentPipelines()[0]?.datasetId).toBe('ds-1');
  });

  it('omits datasetId field when undefined (exactOptionalPropertyTypes)', () => {
    pushRecentPipeline({ name: 'A', nodes: [], edges: [] });
    const entry = getRecentPipelines()[0]!;
    expect('datasetId' in entry).toBe(false);
  });

  it('preserves datasetName when provided and omits it otherwise', () => {
    pushRecentPipeline({ name: 'A', nodes: [], edges: [], datasetName: 'iris.csv' });
    expect(getRecentPipelines()[0]?.datasetName).toBe('iris.csv');
    pushRecentPipeline({ name: 'B', nodes: [], edges: [] });
    expect('datasetName' in getRecentPipelines()[0]!).toBe(false);
  });

  it('swallows quota errors without throwing', () => {
    const setItemSpy = vi
      .spyOn(Storage.prototype, 'setItem')
      .mockImplementation(() => {
        throw new Error('QuotaExceededError');
      });
    expect(() =>
      pushRecentPipeline({ name: 'A', nodes: [], edges: [] }),
    ).not.toThrow();
    setItemSpy.mockRestore();
  });
});
