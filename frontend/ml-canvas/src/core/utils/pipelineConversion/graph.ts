import type { Edge, Node } from '@xyflow/react';
import type { NodeConfigModel } from '../../api/client';
import { StepType } from '../../constants/stepTypes';
import { isModelSourceType } from './ensemble';

/** Model nodes feeding only ensembles supply recipes rather than standalone jobs. */
export function removeSpecOnlyModels(configs: NodeConfigModel[], nodes: Node[], edges: Edge[]): NodeConfigModel[] {
  const ensembleIds = new Set(nodes.filter(node => node.data.definitionType === 'EnsembleNode').map(node => node.id));
  if (ensembleIds.size === 0) return configs;
  return configs.filter(config => {
    const source = nodes.find(node => node.id === config.node_id);
    if (!source || !isModelSourceType(source.data.definitionType)) return true;
    const outgoing = edges.filter(edge => edge.source === config.node_id);
    return outgoing.length === 0 || !outgoing.every(edge => ensembleIds.has(edge.target));
  });
}

/** Keep data leaves alongside explicit terminals, including parallel preview branches. */
function findSeeds(configs: NodeConfigModel[]): NodeConfigModel[] {
  const terminalTypes = new Set([StepType.TRAINING, 'data_preview']);
  const seeds = configs.filter(config => terminalTypes.has(config.step_type));
  if (configs.length <= 1) return seeds;
  const consumed = new Set(configs.flatMap(config => config.inputs));
  const leaves = configs.filter(config => !consumed.has(config.node_id));
  if (seeds.length === 0) return leaves.length > 1 ? leaves : seeds;
  const seedIds = new Set(seeds.map(config => config.node_id));
  return [...seeds, ...leaves.filter(leaf => !seedIds.has(leaf.node_id))];
}

/** Reverse-walk inputs without changing the original dataset traversal order. */
export function pruneToTerminalAncestors(configs: NodeConfigModel[]): NodeConfigModel[] {
  const seeds = findSeeds(configs);
  if (seeds.length === 0) return configs;
  const reachable = new Set<string>();
  const queue = seeds.map(node => node.node_id);
  while (queue.length > 0) {
    const id = queue.shift()!;
    if (reachable.has(id)) continue;
    reachable.add(id);
    const config = configs.find(node => node.node_id === id);
    if (config?.inputs) {
      for (const input of config.inputs) queue.push(input);
    }
  }
  return configs.filter(node => reachable.has(node.node_id));
}
