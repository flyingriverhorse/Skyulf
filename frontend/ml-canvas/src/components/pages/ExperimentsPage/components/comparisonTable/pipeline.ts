import type { JobInfo } from '../../../../../core/api/jobs';
import type { GraphNode, PipelineData, PipelineRow } from './types';
import { summarizeStep } from './pipelineSummary';

/** Traverse parents first and omit the terminal, visiting shared/cyclic nodes once. */
function collectChain(job: JobInfo): GraphNode[] {
  const graphNodes = (job.graph?.nodes as GraphNode[] | undefined) || [];
  if (graphNodes.length === 0 || !job.node_id) return [];
  const nodes = new Map(graphNodes.map(node => [node.node_id, node]));
  const seen = new Set<string>();
  const order: GraphNode[] = [];
  const walk = (id: string): void => {
    if (seen.has(id)) return;
    seen.add(id);
    const node = nodes.get(id);
    if (!node) return;
    for (const parent of node.inputs || []) walk(parent);
    order.push(node);
  };
  walk(job.node_id);
  return order.filter(node => node.node_id !== job.node_id);
}

/** Advance past shared steps and select the next un-emitted node in this chain. */
function nextCandidate(chain: GraphNode[], cursors: number[], index: number, emitted: Set<string>): GraphNode | undefined {
  while (cursors[index]! < chain.length && emitted.has(chain[cursors[index]!]!.node_id)) {
    cursors[index] = cursors[index]! + 1;
  }
  return cursors[index]! < chain.length ? chain[cursors[index]!] : undefined;
}

/** Restart the sweep after each emission to preserve the original chain priority. */
function mergeChains(chains: GraphNode[][]): string[] {
  const mergedOrder: string[] = [];
  const emitted = new Set<string>();
  const cursors = chains.map(() => 0);
  const safety = chains.reduce((sum, chain) => sum + chain.length, 0) + 1;
  for (let guard = 0; guard < safety; guard++) {
    let advanced = false;
    for (let index = 0; index < chains.length; index++) {
      const chain = chains[index];
      if (!chain) continue;
      const candidate = nextCandidate(chain, cursors, index, emitted);
      if (candidate && !emitted.has(candidate.node_id)) {
        mergedOrder.push(candidate.node_id);
        emitted.add(candidate.node_id);
        advanced = true;
        break;
      }
    }
    if (!advanced) break;
  }
  return mergedOrder;
}

function alignedRow(nid: string, idx: number, selectedJobs: JobInfo[], chainsByJob: Map<string, GraphNode[]>): PipelineRow {
  const cells = selectedJobs.map(job => {
    const chain = chainsByJob.get(job.job_id) || [];
    const step = chain.find(node => node.node_id === nid);
    return step ? summarizeStep(step) : null;
  });
  const presentCells = cells.filter((cell): cell is string => cell !== null);
  const allPresent = presentCells.length === cells.length;
  const allSame = allPresent && presentCells.every(cell => cell === presentCells[0]);
  return { nid, idx, cells, allSame };
}

export function preparePipelineData(selectedJobs: JobInfo[]): PipelineData {
  const chainsByJob = new Map(selectedJobs.map(job => [job.job_id, collectChain(job)]));
  const chains = Array.from(chainsByJob.values());
  if (chains.every(chain => chain.length === 0)) return { hasSteps: false, rows: [] };
  const rows = mergeChains(chains).map((nid, idx) => alignedRow(nid, idx, selectedJobs, chainsByJob));
  return { hasSteps: true, rows };
}
