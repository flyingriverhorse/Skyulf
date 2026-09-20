// Templates gallery (L3) — verifies every starter template materialises
// into a non-empty graph and that every referenced node type resolves
// against the live NodeRegistry. Catches typos like `train_test_split`
// vs `TrainTestSplitter` before they reach the UI.

import { describe, it, expect, beforeAll } from 'vitest';
import { initializeRegistry } from '../registry/init';
import { registry } from '../registry/NodeRegistry';
import { PIPELINE_TEMPLATES, buildGraphFromTemplate } from './pipelineTemplates';

beforeAll(() => {
  initializeRegistry();
});

describe('pipeline templates', () => {
  it('binds dataset and target without changing the reusable template', () => {
    // Each new canvas must receive the chosen data without leaking it into later loads.
    const template = PIPELINE_TEMPLATES[0]!;
    const graph = buildGraphFromTemplate(template, { datasetId: '42', datasetName: 'Customers', targetColumn: 'churn' });
    expect(graph.nodes.find(n => n.data.definitionType === 'dataset_node')?.data).toMatchObject({ datasetId: '42', datasetName: 'Customers' });
    expect(graph.nodes.find(n => n.data.definitionType === 'TrainTestSplitter')?.data.target_column).toBe('churn');
    expect(buildGraphFromTemplate(template).nodes[0]?.data.datasetId).toBe('');
  });

  it('binds the text feature to cleaning and vectorization without using the label', () => {
    // Text templates require the same explicit feature column at both preprocessing steps.
    const template = PIPELINE_TEMPLATES.find(t => t.category === 'text')!;
    const graph = buildGraphFromTemplate(template, { datasetId: '42', datasetName: 'Messages', targetColumn: 'label', textColumn: 'message' });
    expect(graph.nodes.find(n => n.data.definitionType === 'TextCleaning')?.data.columns).toEqual(['message']);
    expect(graph.nodes.find(n => n.data.definitionType === 'tfidf_vectorizer')?.data.columns).toEqual(['message']);
    expect(graph.nodes.find(n => n.data.definitionType === 'tfidf_vectorizer')?.data.drop_original).toBe(true);
  });
  it('exposes at least one template', () => {
    expect(PIPELINE_TEMPLATES.length).toBeGreaterThan(0);
  });

  for (const tpl of PIPELINE_TEMPLATES) {
    describe(tpl.id, () => {
      it('references only registered node types', () => {
        for (const node of tpl.nodes) {
          expect(registry.get(node.type), `missing type ${node.type}`).toBeTruthy();
        }
      });

      it('builds a non-empty graph with fresh ids', () => {
        const { nodes, edges } = buildGraphFromTemplate(tpl);
        expect(nodes.length).toBe(tpl.nodes.length);
        expect(edges.length).toBe(tpl.edges.length);
        const ids = new Set(nodes.map((n) => n.id));
        expect(ids.size).toBe(nodes.length); // unique
        // Ids should be uuid-suffixed, not the template-local ids.
        for (const n of nodes) {
          expect(n.id).toMatch(/-/);
        }
      });

      it('produces fresh ids on repeated builds', () => {
        const a = buildGraphFromTemplate(tpl);
        const b = buildGraphFromTemplate(tpl);
        const overlap = a.nodes.some((n) => b.nodes.find((m) => m.id === n.id));
        expect(overlap).toBe(false);
      });
    });
  }
});
