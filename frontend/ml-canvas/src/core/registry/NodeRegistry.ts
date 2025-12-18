import { NodeDefinition } from '../types/nodes';

class NodeRegistry {
  private static instance: NodeRegistry;
  private nodes: Map<string, NodeDefinition<any>> = new Map();

  private constructor() {}

  public static getInstance(): NodeRegistry {
    if (!NodeRegistry.instance) {
      NodeRegistry.instance = new NodeRegistry();
    }
    return NodeRegistry.instance;
  }

  public register<TConfig>(definition: NodeDefinition<TConfig>): void {
    if (this.nodes.has(definition.type)) {
      console.warn(`Node type "${definition.type}" is already registered. Overwriting.`);
    }
    this.nodes.set(definition.type, definition as NodeDefinition<any>);
  }

  public get(type: string): NodeDefinition<any> | undefined {
    return this.nodes.get(type);
  }

  public getAll(): NodeDefinition<any>[] {
    return Array.from(this.nodes.values());
  }

  public getByCategory(): Record<string, NodeDefinition<any>[]> {
    const grouped: Record<string, NodeDefinition<any>[]> = {};
    this.nodes.forEach((node) => {
      // node.category is a string, so it can be used as a key.
      // If it's undefined, it will be "undefined" string key.
      // Assuming category is always present based on type definition.
      if (!grouped[node.category]) {
        grouped[node.category] = [];
      }
      grouped[node.category].push(node);
    });
    return grouped;
  }
}

export const registry = NodeRegistry.getInstance();
