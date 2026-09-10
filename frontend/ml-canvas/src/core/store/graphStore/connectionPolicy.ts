import type { Connection, Edge, Node } from '@xyflow/react';
import { toast } from '../../toast';
import { connectionIssue, MODEL_NODE_TYPES } from '../../utils/connectionValidation';

/** Trace model lineage to dataset roots, tolerating repeated or cyclic ancestors. */
function datasetRoots(startId: string, nodes: Node[], edges: Edge[]): Set<string> {
  const roots = new Set<string>();
  const seen = new Set<string>();
  const stack = [startId];
  while (stack.length > 0) {
    const cur = stack.pop()!;
    if (seen.has(cur)) continue;
    seen.add(cur);
    const n = nodes.find((x) => x.id === cur);
    if (n?.data.definitionType === 'dataset_node') {
      roots.add(cur);
      continue;
    }
    for (const e of edges.filter((ed) => ed.target === cur)) stack.push(e.source);
  }
  return roots;
}

/** Only warn when both lineages are known and share no dataset root. */
function hasDisjointDatasetRoots(connection: Connection, nodes: Node[], edges: Edge[]): boolean {
  const sourceRoots = datasetRoots(connection.source, nodes, edges);
  // Existing lineage of the ensemble = roots of everything already wired in.
  const ensembleRoots = new Set<string>();
  for (const e of edges.filter((ed) => ed.target === connection.target)) {
    for (const r of datasetRoots(e.source, nodes, edges)) ensembleRoots.add(r);
  }
  return (
    ensembleRoots.size > 0 &&
    sourceRoots.size > 0 &&
    ![...sourceRoots].some((r) => ensembleRoots.has(r))
  );
}

/** Ensemble model inputs are specifications that will be refitted on ensemble data. */
function confirmEnsembleLineage(connection: Connection, sourceType: string, targetType: string, nodes: Node[], edges: Edge[]): boolean {
  const modelSourceTypes = ['classification', 'regression', 'text_classification', 'SegmentationNode'];
  if (targetType !== 'EnsembleNode' || !modelSourceTypes.includes(sourceType)) return true;
  if (!hasDisjointDatasetRoots(connection, nodes, edges)) return true;
  return window.confirm(
    'Warning: this model comes from a different dataset than the ensemble\'s ' +
    'other inputs.\n\n' +
    'The ensemble re-fits every base learner on a single dataset, so mixing ' +
    'models trained on unrelated data is usually a wiring mistake.\n\n' +
    'Click OK to connect anyway, or Cancel to abort.'
  );
}

/** Follow every upstream branch looking for a train/test boundary. */
function hasUpstreamTrainTestSplit(sourceId: string, nodes: Node[], edges: Edge[]): boolean {
  const queue = [sourceId];
  const visited = new Set<string>();

  while (queue.length > 0) {
    const currentId = queue.shift()!;
    if (visited.has(currentId)) continue;
    visited.add(currentId);

    const currentNode = nodes.find((n) => n.id === currentId);
    if (currentNode?.data.definitionType === 'TrainTestSplitter') {
      return true;
    }

    const parentEdges = edges.filter((e) => e.target === currentId);
    for (const edge of parentEdges) {
      queue.push(edge.source);
    }
  }
  return false;
}

/** Keep this confirmation synchronous so cancellation precedes graph mutation. */
function confirmTrainTestSplit(sourceNode: Node, targetNode: Node, nodes: Node[], edges: Edge[]): boolean {
  if (sourceNode.data.definitionType !== 'feature_target_split') return true;
  if (targetNode.data.definitionType === 'TrainTestSplitter') return true;
  if (hasUpstreamTrainTestSplit(sourceNode.id, nodes, edges)) return true;
  return window.confirm(
    'Warning: X/Y Split without a prior Train-Test Split.\n\n' +
    'This means 100% of data will be used (possible data leakage).\n\n' +
    'Click OK to connect anyway, or Cancel to abort.'
  );
}

/** Training fan-in explains merge versus separate experiments. */
function confirmTrainingFanIn(uniqueSourceCount: number): boolean {
  return window.confirm(
    `This training node will receive ${uniqueSourceCount} inputs.\n\n` +
    'You have two options:\n' +
    '  • MERGE (default): Inputs are auto-merged into one dataset before training.\n' +
    '  • PARALLEL: Each input runs as a separate experiment.\n' +
    '    → To use parallel mode, connect each path to its OWN training node.\n\n' +
    'Click OK to connect (merge mode), or Cancel to abort.'
  );
}

/** Processing fan-in explains column union and deferred conflict resolution. */
function confirmProcessingFanIn(uniqueSourceCount: number): boolean {
  return window.confirm(
    `This node will receive ${uniqueSourceCount} inputs.\n\n` +
    'Inputs are merged into one dataset. Each column keeps the value of ' +
    'whichever branch changed it, so branches editing different columns ' +
    'are combined without loss.\n\n' +
    'If two branches change the SAME column to different values, one of ' +
    'them is discarded. The run Results panel reports any such conflict ' +
    'and lets you choose which branch wins.\n\n' +
    'For strictly sequential transformations, chain the nodes linearly instead.\n\n' +
    'Click OK to connect (merge), or Cancel to abort.'
  );
}

/** Count logical source nodes rather than their individual output handles. */
function confirmFanIn(connection: Connection, targetType: string, edges: Edge[]): boolean {
  const existingInputs = edges.filter(e => e.target === connection.target);
  const existingSources = new Set(existingInputs.map(e => e.source));
  const isNewSource = connection.source != null && !existingSources.has(connection.source);
  const uniqueSourceCount = existingSources.size + (isNewSource ? 1 : 0);
  // Preview tabs and ensemble specifications have their own input contracts.
  if (targetType === 'data_preview' || targetType === 'EnsembleNode') return true;
  if (!isNewSource || uniqueSourceCount < 2) return true;
  return MODEL_NODE_TYPES.includes(targetType)
    ? confirmTrainingFanIn(uniqueSourceCount)
    : confirmProcessingFanIn(uniqueSourceCount);
}

/** Preserve ordered synchronous preflight for wiring and atomic next-step insertion. */
export function confirmConnection(connection: Connection, nodes: Node[], edges: Edge[]): boolean {
  const issue = connectionIssue(nodes, edges, connection);
  if (issue) {
    toast.error('Invalid connection', issue);
    return false;
  }
  const sourceNode = nodes.find(n => n.id === connection.source);
  const targetNode = nodes.find(n => n.id === connection.target);
  if (!sourceNode || !targetNode) return true;
  const sourceType = sourceNode.data.definitionType as string;
  const targetType = targetNode.data.definitionType as string;
  if (!confirmEnsembleLineage(connection, sourceType, targetType, nodes, edges)) return false;
  if (!confirmTrainTestSplit(sourceNode, targetNode, nodes, edges)) return false;
  return confirmFanIn(connection, targetType, edges);
}
