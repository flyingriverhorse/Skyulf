import { create } from 'zustand';
import { temporal } from 'zundo';
import type { TemporalState } from 'zundo';
import { useStore } from 'zustand';
import {
  Connection,
  Edge,
  EdgeChange,
  Node,
  NodeChange,
  addEdge,
  OnNodesChange,
  OnEdgesChange,
  OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
} from '@xyflow/react';
import { registry } from '../registry/NodeRegistry';
import { PreviewResponse } from '../api/client';
import type { NodeSummaryEntry } from '../api/jobs';
import { v4 as uuidv4 } from 'uuid';
import {
  type ExecutionMode,
  getExecutionMode as readExecutionMode,
} from '../types/executionMode';
import { toast } from '../toast';

interface GraphState {
  nodes: Node[];
  edges: Edge[];
  
  // React Flow Actions
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  
  // Custom Actions
  addNode: (type: string, position: { x: number, y: number }, initialData?: unknown) => string;
  updateNodeData: (id: string, data: unknown) => void;
  /**
   * Read the effective `execution_mode` for a node, defaulting to the
   * canonical `'merge'` when unset. Cheaper than `useGraphStore` callers
   * pulling `nodes` and re-deriving the value, and keeps the cast to
   * `ExecutionModeData` in one place.
   */
  getExecutionMode: (nodeId: string) => ExecutionMode;
  /** Typed setter for the modeling-node Merge/Parallel toggle. */
  setExecutionMode: (nodeId: string, mode: ExecutionMode) => void;
  /**
   * Clone every currently-selected node with a small position offset
   * and a fresh id. New clones become the selection (originals are
   * deselected). Edges between selected nodes are NOT copied — keep
   * it predictable; users can re-wire if they want a parallel branch.
   * Returns the count of cloned nodes (0 when nothing is selected).
   */
  duplicateSelectedNodes: () => number;
  validateGraph: () => Promise<boolean>;
  setGraph: (nodes: Node[], edges: Edge[]) => void;
  /**
   * Rewire a sibling fan-in into a linear chain.
   *
   * Given inputs `[A, B, C]` all feeding `consumerId`, this:
   *  1. Removes edges `A→consumer`, `B→consumer`, `C→consumer`.
   *  2. Adds edges `A→B`, `B→C`, `C→consumer` (using each node's first
   *     input handle and source's first output handle).
   *
   * No cycle check is needed because the new edges only go between nodes
   * that previously fed `consumerId` (so they're already topologically
   * before it). Returns `true` on success, `false` if validation rejects
   * the change.
   */
  chainSiblings: (consumerId: string, orderedInputIds: string[]) => boolean;

  // Execution State
  executionResult: PreviewResponse | null;
  setExecutionResult: (result: PreviewResponse | null) => void;

  // Per-node card summaries sourced from completed training/tuning
  // jobs. Trainer/tuner jobs run via Celery and the engine's per-node
  // `metadata.summary` never makes it into `executionResult` (the
  // `/preview` path strips trainers before execution). The
  // `useNodeJobSummaries` hook polls `/pipeline/jobs/node-summaries`
  // and writes the result here so trainer cards can fall back to it.
  // For parallel terminals each branch contributes one entry so the
  // card can render Path A / Path B / … on separate lines.
  nodeJobSummaries: Record<string, NodeSummaryEntry[]>;
  setNodeJobSummaries: (summaries: Record<string, NodeSummaryEntry[]>) => void;

  // Canvas-derived map: edgeId -> Path label (e.g. "Path B · Xgboost").
  // Written by `FlowCanvas` after `useBranchColors` runs so other
  // surfaces (trainer cards) can show the same Path letters the user
  // sees on the canvas without re-deriving the partition logic.
  branchEdgeLabels: Record<string, string>;
  setBranchEdgeLabels: (labels: Record<string, string>) => void;
}

export const useGraphStore = create<GraphState>()(
  temporal(
    (set, get) => ({
  nodes: [],
  edges: [],
  executionResult: null,

  setExecutionResult: (result) => set({ executionResult: result }),

  nodeJobSummaries: {},
  setNodeJobSummaries: (summaries) => set({ nodeJobSummaries: summaries }),

  branchEdgeLabels: {},
  setBranchEdgeLabels: (labels) => set({ branchEdgeLabels: labels }),
  setGraph: (nodes, edges) => set({ nodes, edges }),

  onNodesChange: (changes: NodeChange[]) => {
    set({
      nodes: applyNodeChanges(changes, get().nodes),
    });
  },

  onEdgesChange: (changes: EdgeChange[]) => {
    set({
      edges: applyEdgeChanges(changes, get().edges),
    });
  },

  onConnect: (connection: Connection) => {
    const nodes = get().nodes;
    const edges = get().edges;
    const sourceNode = nodes.find((n) => n.id === connection.source);
    const targetNode = nodes.find((n) => n.id === connection.target);

    if (sourceNode && targetNode) {
      const sourceType = sourceNode.data.definitionType as string;
      const targetType = targetNode.data.definitionType as string;

      // Block: connecting a training/tuning output into another training/tuning node
      const modelTypes = ['basic_training', 'advanced_tuning'];
      if (modelTypes.includes(sourceType) && modelTypes.includes(targetType)) {
        toast.error(
          'Invalid connection',
          'You cannot connect a model output to another training node. Training nodes expect data (DataFrame), not a trained model.',
        );
        return;
      }

      // Warn: X/Y Split without prior Train-Test Split
      if (sourceType === 'feature_target_split') {
        let hasTrainTestSplit = targetNode.data.definitionType === 'TrainTestSplitter';
        
        if (!hasTrainTestSplit) {
          const queue = [sourceNode.id];
          const visited = new Set<string>();

          while (queue.length > 0) {
            const currentId = queue.shift()!;
            if (visited.has(currentId)) continue;
            visited.add(currentId);

            const currentNode = nodes.find((n) => n.id === currentId);
            if (currentNode?.data.definitionType === 'TrainTestSplitter') {
              hasTrainTestSplit = true;
              break;
            }

            const parentEdges = edges.filter((e) => e.target === currentId);
            for (const edge of parentEdges) {
              queue.push(edge.source);
            }
          }
        }

        if (!hasTrainTestSplit) {
          // window.confirm is intentional here: onConnect is a synchronous
          // React Flow callback that must return before the edge is
          // committed, so we can't await the async <ConfirmDialog>.
          const proceed = window.confirm(
            'Warning: X/Y Split without a prior Train-Test Split.\n\n' +
            'This means 100% of data will be used (possible data leakage).\n\n' +
            'Click OK to connect anyway, or Cancel to abort.'
          );
          if (!proceed) return;
        }
      }

      // Warn: multi-input on a training/tuning node — explain merge vs parallel.
      // Count UNIQUE source nodes, not edges: multi-output splitters
      // (train_test_split, feature_target_split) legitimately produce
      // several edges from the same source (train/test/X/y handles)
      // into one downstream node, and that's not fan-in — the engine
      // dedupes by source id, so no merge happens.
      const existingInputs = edges.filter(e => e.target === connection.target);
      const existingSources = new Set(existingInputs.map(e => e.source));
      const isNewSource = connection.source != null && !existingSources.has(connection.source);
      const uniqueSourceCount = existingSources.size + (isNewSource ? 1 : 0);
      // Auto-parallel terminals (data_preview) split each input into its own
      // tab instead of merging — no warning needed, no merge contract to
      // confirm. Mirrors AUTO_PARALLEL_STEP_TYPES in backend graph_utils.py.
      const autoParallelTypes = ['data_preview'];
      if (autoParallelTypes.includes(targetType)) {
        // Skip both confirms below; each input becomes its own preview tab.
      } else if (isNewSource && uniqueSourceCount >= 2 && modelTypes.includes(targetType)) {
        // window.confirm: see note above on sync onConnect contract.
        const proceed = window.confirm(
          `This training node will receive ${uniqueSourceCount} inputs.\n\n` +
          'You have two options:\n' +
          '  • MERGE (default): Inputs are auto-merged into one dataset before training.\n' +
          '  • PARALLEL: Each input runs as a separate experiment.\n' +
          '    → To use parallel mode, connect each path to its OWN training node.\n\n' +
          'Click OK to connect (merge mode), or Cancel to abort.'
        );
        if (!proceed) return;
      } else if (isNewSource && uniqueSourceCount >= 2 && !modelTypes.includes(targetType)) {
        // Pre-flight lint for non-training nodes (audit issue #7).
        // Direction-A means non-training nodes also auto-merge fan-in via
        // column union + last-wins. Surface that contract before the user
        // wires it so silent column overwrites aren't a surprise at run time.
        // window.confirm: see note above on sync onConnect contract.
        const proceed = window.confirm(
          `This node will receive ${uniqueSourceCount} inputs.\n\n` +
          'Inputs are auto-merged via column union with LAST-WINS on overlap ' +
          '(the last connected input overwrites earlier ones on shared columns).\n\n' +
          'For sequential transformations, chain the nodes linearly instead.\n\n' +
          'Click OK to connect (merge), or Cancel to abort.'
        );
        if (!proceed) return;
      }
    }

    set({
      edges: addEdge(connection, get().edges),
    });
  },

  addNode: (type: string, position: { x: number, y: number }, initialData: unknown = {}) => {
    const definition = registry.get(type);
    if (!definition) {
      console.error(`Node type ${type} not found in registry`);
      return '';
    }

    const id = `${type}-${uuidv4()}`;
    const newNode: Node = {
      id,
      type: 'custom', // We'll use a generic wrapper component
      position,
      data: { 
        // Store the definition type so we can look it up later
        definitionType: type,
        catalogType: type, // For backend execution compatibility
        // Initialize with default values if any
        ...(definition.getDefaultConfig() as object),
        ...(initialData as object)
      },
    };

    set({ nodes: [...get().nodes, newNode] });
    return id;
  },

  updateNodeData: (id: string, data: unknown) => {
    set({
      nodes: get().nodes.map((node) => {
        if (node.id === id) {
          return { ...node, data: { ...node.data, ...(data as object) } };
        }
        return node;
      }),
    });
  },

  getExecutionMode: (nodeId: string) => {
    const node = get().nodes.find((n) => n.id === nodeId);
    return readExecutionMode(node?.data as { execution_mode?: ExecutionMode } | undefined);
  },

  setExecutionMode: (nodeId: string, mode: ExecutionMode) => {
    set({
      nodes: get().nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, execution_mode: mode } }
          : node,
      ),
    });
  },

  duplicateSelectedNodes: () => {
    const current = get().nodes;
    const selected = current.filter((n) => n.selected);
    if (selected.length === 0) return 0;
    const OFFSET = 32;
    const clones: Node[] = selected.map((src) => ({
      ...src,
      id: `${src.data.definitionType as string}-${uuidv4()}`,
      position: { x: src.position.x + OFFSET, y: src.position.y + OFFSET },
      selected: true,
      data: { ...src.data },
    }));
    const deselected = current.map((n) =>
      n.selected ? { ...n, selected: false } : n
    );
    set({ nodes: [...deselected, ...clones] });
    return clones.length;
  },

  chainSiblings: (consumerId: string, orderedInputIds: string[]) => {
    const inputs = orderedInputIds.filter(Boolean);
    if (inputs.length < 2) return false;

    const { edges, nodes } = get();
    if (!nodes.find((n) => n.id === consumerId)) return false;

    const firstOutputHandle = (nodeId: string): string | undefined => {
      const node = nodes.find((n) => n.id === nodeId);
      const def = node && registry.get(node.data.definitionType as string);
      return def?.outputs?.[0]?.id;
    };
    const firstInputHandle = (nodeId: string): string | undefined => {
      const node = nodes.find((n) => n.id === nodeId);
      const def = node && registry.get(node.data.definitionType as string);
      return def?.inputs?.[0]?.id;
    };

    const links: Array<{ source: string; target: string; sourceHandle?: string; targetHandle?: string }> = [];
    for (let i = 0; i < inputs.length - 1; i++) {
      const a = inputs[i]!;
      const b = inputs[i + 1]!;
      const sh = firstOutputHandle(a);
      const th = firstInputHandle(b);
      if (!sh || !th) return false;
      links.push({ source: a, target: b, sourceHandle: sh, targetHandle: th });
    }
    const lastInput = inputs[inputs.length - 1]!;
    const lastSh = firstOutputHandle(lastInput);
    const lastTh = firstInputHandle(consumerId);
    if (!lastSh || !lastTh) return false;
    links.push({ source: lastInput, target: consumerId, sourceHandle: lastSh, targetHandle: lastTh });

    // Keep head sibling's upstream edges; clear incoming on tail siblings
    // and drop the original fan-in into the consumer.
    const inputSet = new Set(inputs);
    const tailSiblings = new Set(inputs.slice(1));
    let nextEdges = edges.filter((e) => {
      if (e.target === consumerId && inputSet.has(e.source)) return false;
      if (tailSiblings.has(e.target)) return false;
      return true;
    });

    for (const link of links) {
      nextEdges = addEdge(link as Connection, nextEdges);
    }

    set({ edges: nextEdges });
    return true;
  },

  validateGraph: async () => {
    const { nodes, edges } = get();
    
    // 1. Check if graph is empty
    if (nodes.length === 0) {
      return false;
    }

    // 2. Run individual node validation
    for (const node of nodes) {
      const definition = registry.get(node.data.definitionType as string);
      if (definition) {
        const validation = definition.validate(node.data);
        if (!validation.isValid) {
          // TODO: We could store this error in the node state to show a visual indicator
          console.warn(`Node ${node.id} validation failed: ${validation.message}`);
          return false;
        }
      }
    }

    // 3. Check connectivity (Basic check: All nodes except sources must have at least one input)
    // This is a heuristic; some nodes might be optional or standalone, but in a pipeline, usually everything connects.
    // We can refine this by checking definition.inputs.length > 0
    for (const node of nodes) {
      const definition = registry.get(node.data.definitionType as string);
      if (definition && definition.inputs.length > 0) {
        const hasInput = edges.some(e => e.target === node.id);
        if (!hasInput) {
          console.warn(`Node ${node.id} (${definition.label}) is disconnected.`);
          return false;
        }
      }
    }

    return true;
  },
    }),
    {
      // Only track structural graph state for undo/redo. Excluding
      // `executionResult` keeps preview data out of history (it's
      // populated by the backend, not user actions, and snapshots
      // of it can be tens of MB).
      partialize: (state) => ({ nodes: state.nodes, edges: state.edges }),
      // Skip history entries that only differ by an in-progress drag.
      // React Flow emits a stream of `dragging:true` position changes
      // per pointer move; we only care about the committed final
      // position (when `dragging` flips to false). Same for plain
      // selection changes — toggling `selected` shouldn't be undoable.
      equality: (prev, next) => {
        if (prev.edges !== next.edges) return false;
        if (prev.nodes === next.nodes) return true;
        if (prev.nodes.length !== next.nodes.length) return false;
        for (let i = 0; i < prev.nodes.length; i++) {
          const a = prev.nodes[i]!;
          const b = next.nodes[i]!;
          if (a.id !== b.id) return false;
          if (a.data !== b.data) return false;
          if (a.type !== b.type) return false;
          // Treat any node currently being dragged as equal to its
          // previous state — only the drag-end commit creates a
          // history entry.
          if (a.dragging || b.dragging) continue;
          if (a.position.x !== b.position.x || a.position.y !== b.position.y) return false;
        }
        return true;
      },
      limit: 100,
    },
  ),
);

/**
 * Hook into the temporal substore (undo/redo). Use selectors to
 * subscribe to specific slices, e.g.
 *   const undo = useTemporalStore((s) => s.undo);
 *   const canUndo = useTemporalStore((s) => s.pastStates.length > 0);
 */
export const useTemporalStore = <T,>(
  selector: (state: TemporalState<{ nodes: Node[]; edges: Edge[] }>) => T,
): T => useStore(useGraphStore.temporal, selector);
