import { describe, it, expect, beforeEach } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import {
  saveCanvasSnapshot,
  loadCanvasSnapshot,
  clearCanvasSnapshot,
} from './canvasPersistence';

const LS_KEY = 'skyulf:canvas:autosave:v1';

const sampleNodes: Node[] = [
  { id: 'a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } },
];
const sampleEdges: Edge[] = [{ id: 'a-b', source: 'a', target: 'b' }];

describe('canvasPersistence', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('round-trips a snapshot through save → load', () => {
    saveCanvasSnapshot(sampleNodes, sampleEdges);
    const snap = loadCanvasSnapshot();
    expect(snap).not.toBeNull();
    expect(snap?.nodes).toEqual(sampleNodes);
    expect(snap?.edges).toEqual(sampleEdges);
    expect(snap?.version).toBe(1);
    // savedAt is an ISO timestamp from the save call.
    expect(new Date(snap!.savedAt).toString()).not.toBe('Invalid Date');
  });

  it('returns null when nothing has been saved', () => {
    expect(loadCanvasSnapshot()).toBeNull();
  });

  it('returns null on a corrupt JSON payload (no crash)', () => {
    window.localStorage.setItem(LS_KEY, '{not json');
    expect(loadCanvasSnapshot()).toBeNull();
  });

  it('returns null when the schema version does not match', () => {
    window.localStorage.setItem(
      LS_KEY,
      JSON.stringify({ version: 999, savedAt: new Date().toISOString(), nodes: [], edges: [] }),
    );
    expect(loadCanvasSnapshot()).toBeNull();
  });

  it('returns null when nodes/edges are missing or malformed', () => {
    window.localStorage.setItem(
      LS_KEY,
      JSON.stringify({ version: 1, savedAt: new Date().toISOString() }),
    );
    expect(loadCanvasSnapshot()).toBeNull();
  });

  it('clearCanvasSnapshot removes the stored payload', () => {
    saveCanvasSnapshot(sampleNodes, sampleEdges);
    expect(loadCanvasSnapshot()).not.toBeNull();
    clearCanvasSnapshot();
    expect(loadCanvasSnapshot()).toBeNull();
    expect(window.localStorage.getItem(LS_KEY)).toBeNull();
  });
});
