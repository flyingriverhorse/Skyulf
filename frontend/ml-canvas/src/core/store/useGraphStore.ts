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

    const id = `${type}-${Date.now()}`;
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
