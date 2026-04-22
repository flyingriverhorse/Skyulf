import { create } from 'zustand';
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
import { v4 as uuidv4 } from 'uuid';

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
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  executionResult: null,

  setExecutionResult: (result) => set({ executionResult: result }),
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
        alert(
          'Invalid connection: You cannot connect a model output to another training node.\n\n' +
          'Training nodes expect data (DataFrame), not a trained model.'
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
          const proceed = confirm(
            'Warning: X/Y Split without a prior Train-Test Split.\n\n' +
            'This means 100% of data will be used (possible data leakage).\n\n' +
            'Click OK to connect anyway, or Cancel to abort.'
          );
          if (!proceed) return;
        }
      }

      // Warn: multi-input on a training/tuning node — explain merge vs parallel
      const existingInputs = edges.filter(e => e.target === connection.target);
      if (existingInputs.length >= 1 && modelTypes.includes(targetType)) {
        const inputCount = existingInputs.length + 1;
        const proceed = confirm(
          `This training node will receive ${inputCount} inputs.\n\n` +
          'You have two options:\n' +
          '  • MERGE (default): Inputs are auto-merged into one dataset before training.\n' +
          '  • PARALLEL: Each input runs as a separate experiment.\n' +
          '    → To use parallel mode, connect each path to its OWN training node.\n\n' +
          'Click OK to connect (merge mode), or Cancel to abort.'
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
        ...definition.getDefaultConfig(),
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
      const sh = firstOutputHandle(inputs[i]);
      const th = firstInputHandle(inputs[i + 1]);
      if (!sh || !th) return false;
      links.push({ source: inputs[i], target: inputs[i + 1], sourceHandle: sh, targetHandle: th });
    }
    const lastSh = firstOutputHandle(inputs[inputs.length - 1]);
    const lastTh = firstInputHandle(consumerId);
    if (!lastSh || !lastTh) return false;
    links.push({ source: inputs[inputs.length - 1], target: consumerId, sourceHandle: lastSh, targetHandle: lastTh });

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
}));
