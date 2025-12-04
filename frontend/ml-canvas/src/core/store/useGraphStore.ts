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

interface GraphState {
  nodes: Node[];
  edges: Edge[];
  
  // React Flow Actions
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  
  // Custom Actions
  addNode: (type: string, position: { x: number, y: number }, initialData?: any) => string;
  updateNodeData: (id: string, data: any) => void;
  validateGraph: () => Promise<boolean>;
  setGraph: (nodes: Node[], edges: Edge[]) => void;

  // Execution State
  executionResult: any | null;
  setExecutionResult: (result: any) => void;
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

  addNode: (type: string, position: { x: number, y: number }, initialData: any = {}) => {
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
        ...initialData
      },
    };

    set({ nodes: [...get().nodes, newNode] });
    return id;
  },

  updateNodeData: (id: string, data: any) => {
    set({
      nodes: get().nodes.map((node) => {
        if (node.id === id) {
          return { ...node, data: { ...node.data, ...data } };
        }
        return node;
      }),
    });
  },

  validateGraph: async () => {
    // TODO: Implement graph validation logic
    // 1. Check if all required ports are connected
    // 2. Run individual node validation
    return true;
  },
}));
