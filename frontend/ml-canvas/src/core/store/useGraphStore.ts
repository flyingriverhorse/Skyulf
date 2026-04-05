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
