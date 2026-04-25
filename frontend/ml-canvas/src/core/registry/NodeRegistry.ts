import { NodeDefinition } from '../types/nodes';

// Heterogeneous registry: each entry has its own TConfig, so we use `unknown`
// as the storage shape and let `register<TConfig>` accept the typed signature.
type AnyNodeDefinition = NodeDefinition<unknown>;

class NodeRegistry {
  private static instance: NodeRegistry;
  private nodes: Map<string, AnyNodeDefinition> = new Map();

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
    this.nodes.set(definition.type, definition as unknown as AnyNodeDefinition);
  }

  public get(type: string): AnyNodeDefinition | undefined {
    return this.nodes.get(type);
  }

  public getAll(): AnyNodeDefinition[] {
    return Array.from(this.nodes.values());
  }

  public getByCategory(): Record<string, AnyNodeDefinition[]> {
    const grouped: Record<string, AnyNodeDefinition[]> = {};
    this.nodes.forEach((node) => {
      const bucket = grouped[node.category] ?? [];
      bucket.push(node);
      grouped[node.category] = bucket;
    });
    return grouped;
  }
}

export const registry = NodeRegistry.getInstance();
