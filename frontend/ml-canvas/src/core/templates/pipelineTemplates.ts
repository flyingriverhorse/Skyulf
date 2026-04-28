/**
 * Starter pipeline templates (L3).
 *
 * Each template is a curated graph users can drop onto the canvas
 * with one click instead of wiring 8+ nodes by hand. We deliberately
 * do not bind a dataset — every template includes a `dataset_node`
 * placeholder the user must point at their own data before running.
 *
 * Templates are pure data + a tiny `buildGraph` factory so loading
 * always produces fresh ids (avoids React Flow id collisions when
 * the user instantiates the same template twice).
 */

import type { Edge, Node } from '@xyflow/react';
import { v4 as uuidv4 } from 'uuid';
import { registry } from '../registry/NodeRegistry';
import { Layers, LineChart, FileText } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

/** A node entry inside a template — just enough to materialise via `addNode`. */
interface TemplateNode {
  /** Template-local id — referenced by edges below. Not used at runtime. */
  localId: string;
  /** Definition type registered in the NodeRegistry. */
  type: string;
  position: { x: number; y: number };
  /** Optional overrides on top of `definition.getDefaultConfig()`. */
  data?: Record<string, unknown>;
}

interface TemplateEdge {
  source: string; // localId
  target: string; // localId
  sourceHandle?: string;
  targetHandle?: string;
}

export interface PipelineTemplate {
  id: string;
  name: string;
  description: string;
  category: 'classification' | 'regression' | 'forecast' | 'text';
  icon: LucideIcon;
  nodes: TemplateNode[];
  edges: TemplateEdge[];
}

/**
 * Materialise a template into React Flow `nodes` / `edges` with fresh
 * UUIDs. Falls back to an empty payload if any node type is missing
 * from the registry (defensive — keeps loading from throwing if a
 * template references a node that has been renamed).
 */
export function buildGraphFromTemplate(template: PipelineTemplate): {
  nodes: Node[];
  edges: Edge[];
} {
  const idMap = new Map<string, string>();
  const nodes: Node[] = [];
  for (const t of template.nodes) {
    const definition = registry.get(t.type);
    if (!definition) continue;
    const realId = `${t.type}-${uuidv4()}`;
    idMap.set(t.localId, realId);
    nodes.push({
      id: realId,
      type: 'custom',
      position: t.position,
      data: {
        definitionType: t.type,
        catalogType: t.type,
        ...(definition.getDefaultConfig() as object),
        ...(t.data ?? {}),
      },
    });
  }
  const edges: Edge[] = [];
  for (const e of template.edges) {
    const source = idMap.get(e.source);
    const target = idMap.get(e.target);
    if (!source || !target) continue;
    edges.push({
      id: `e-${source}-${target}-${uuidv4().slice(0, 8)}`,
      source,
      target,
      ...(e.sourceHandle ? { sourceHandle: e.sourceHandle } : {}),
      ...(e.targetHandle ? { targetHandle: e.targetHandle } : {}),
    });
  }
  return { nodes, edges };
}

// ---------------------------------------------------------------------------
// Template catalog
// ---------------------------------------------------------------------------

// Layout helpers — keep nodes spaced consistently across templates so
// the canvas reads left-to-right without overlap on first paint.
const COL = 280;
const ROW = 140;
const col = (i: number): number => 80 + i * COL;
const row = (i: number): number => 80 + i * ROW;

const TABULAR_CLASSIFICATION: PipelineTemplate = {
  id: 'tabular_classification',
  name: 'Tabular Classification',
  description:
    'Classic supervised pipeline: dataset → drop ids → impute → encode → scale → split → train. Bind your dataset, set the target column on the split node, and Run All.',
  category: 'classification',
  icon: Layers,
  nodes: [
    { localId: 'ds', type: 'dataset_node', position: { x: col(0), y: row(0) } },
    { localId: 'drop', type: 'drop_missing_columns', position: { x: col(1), y: row(0) } },
    { localId: 'imp', type: 'imputation_node', position: { x: col(2), y: row(0) } },
    { localId: 'enc', type: 'encoding', position: { x: col(3), y: row(0) } },
    { localId: 'scl', type: 'scale_numeric_features', position: { x: col(4), y: row(0) } },
    { localId: 'split', type: 'TrainTestSplitter', position: { x: col(5), y: row(0) } },
    { localId: 'train', type: 'basic_training', position: { x: col(6), y: row(0) } },
  ],
  edges: [
    { source: 'ds', target: 'drop' },
    { source: 'drop', target: 'imp' },
    { source: 'imp', target: 'enc' },
    { source: 'enc', target: 'scl' },
    { source: 'scl', target: 'split' },
    { source: 'split', target: 'train' },
  ],
};

const TABULAR_REGRESSION: PipelineTemplate = {
  id: 'tabular_regression',
  name: 'Tabular Regression',
  description:
    'Regression starter: dataset → impute → outlier removal → scale → split → train. Pick a numeric target column on the split node before running.',
  category: 'regression',
  icon: LineChart,
  nodes: [
    { localId: 'ds', type: 'dataset_node', position: { x: col(0), y: row(0) } },
    { localId: 'imp', type: 'imputation_node', position: { x: col(1), y: row(0) } },
    { localId: 'out', type: 'outlier', position: { x: col(2), y: row(0) } },
    { localId: 'scl', type: 'scale_numeric_features', position: { x: col(3), y: row(0) } },
    { localId: 'split', type: 'TrainTestSplitter', position: { x: col(4), y: row(0) } },
    {
      localId: 'train',
      type: 'basic_training',
      position: { x: col(5), y: row(0) },
      data: { model_type: 'linear_regression' },
    },
  ],
  edges: [
    { source: 'ds', target: 'imp' },
    { source: 'imp', target: 'out' },
    { source: 'out', target: 'scl' },
    { source: 'scl', target: 'split' },
    { source: 'split', target: 'train' },
  ],
};

const TEXT_CLASSIFICATION: PipelineTemplate = {
  id: 'text_classification',
  name: 'Text Classification',
  description:
    'Text starter: dataset → text cleaning → encoding → split → train. Wire the Text Cleaning node at the column with your raw text, and pick the label column on the split.',
  category: 'text',
  icon: FileText,
  nodes: [
    { localId: 'ds', type: 'dataset_node', position: { x: col(0), y: row(0) } },
    { localId: 'clean', type: 'TextCleaning', position: { x: col(1), y: row(0) } },
    { localId: 'enc', type: 'encoding', position: { x: col(2), y: row(0) } },
    { localId: 'split', type: 'TrainTestSplitter', position: { x: col(3), y: row(0) } },
    { localId: 'train', type: 'basic_training', position: { x: col(4), y: row(0) } },
  ],
  edges: [
    { source: 'ds', target: 'clean' },
    { source: 'clean', target: 'enc' },
    { source: 'enc', target: 'split' },
    { source: 'split', target: 'train' },
  ],
};

export const PIPELINE_TEMPLATES: PipelineTemplate[] = [
  TABULAR_CLASSIFICATION,
  TABULAR_REGRESSION,
  TEXT_CLASSIFICATION,
];
