import { NodeDefinition } from '../types/nodes';

class NodeRegistry {
  private static instance: NodeRegistry;
  private nodes: Map<string, NodeDefinition> = new Map();

  private constructor() {}

  public static getInstance(): NodeRegistry {
    if (!NodeRegistry.instance) {
      NodeRegistry.instance = new NodeRegistry();
    }
    return NodeRegistry.instance;
  }

  public register(definition: NodeDefinition): void {
    if (this.nodes.has(definition.type)) {
      console.warn(`Node type "${definition.type}" is already registered. Overwriting.`);
    }
    this.nodes.set(definition.type, definition);
    console.log(`[Registry] Registered node: ${definition.label} (${definition.type})`);
  }

  public get(type: string): NodeDefinition | undefined {
    return this.nodes.get(type);
  }

  public getAll(): NodeDefinition[] {
    return Array.from(this.nodes.values());
  }

  public getByCategory(): Record<string, NodeDefinition[]> {
    const grouped: Record<string, NodeDefinition[]> = {};
    this.nodes.forEach((node) => {
      if (!grouped[node.category]) {
        grouped[node.category] = [];
      }
      grouped[node.category].push(node);
    });
    return grouped;
  }
}

export const registry = NodeRegistry.getInstance();
