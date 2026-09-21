// Templates gallery (L3) — verifies every starter template materialises
// into a non-empty graph and that every referenced node type resolves
// against the live NodeRegistry. Catches typos like `train_test_split`
// vs `TrainTestSplitter` before they reach the UI.

import { describe, it, expect, beforeAll } from 'vitest';
import { initializeRegistry } from '../registry/init';
import { registry } from '../registry/NodeRegistry';
import { convertGraphToPipelineConfig } from '../utils/pipelineConverter';
import { PIPELINE_TEMPLATES, buildGraphFromTemplate } from './pipelineTemplates';

beforeAll(() => {
  initializeRegistry();
});

describe('pipeline templates', () => {
  it.each(PIPELINE_TEMPLATES.filter(template => template.category !== 'clustering'))(
    'binds the selected target through model validation and conversion for $id', template => {
      // Preprocessing between the splitter and model must not require selecting the target again.
      const binding = { datasetId: '42', datasetName: 'Customers', targetColumn: 'label', textColumn: 'message' };
      const { nodes, edges } = buildGraphFromTemplate(template, binding);
      const model = nodes[nodes.length - 1]!;
      expect(model.data.target_column).toBe('label');
      expect(registry.get(String(model.data.definitionType))!.validate(model.data).isValid).toBe(true);
      const converted = convertGraphToPipelineConfig(nodes, edges);
      expect(converted.nodes.find(node => node.node_id === model.id)?.params.target_column).toBe('label');
      expect(buildGraphFromTemplate(template).nodes.at(-1)?.data.target_column).toBe('');
    },
  );

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
