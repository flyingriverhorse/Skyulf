import type { Node } from '@xyflow/react';
import { registry } from '../registry/NodeRegistry';

/** Give connections readable endpoint names, with ordinals for repeated labels. */
export function nodeDisplayNames(nodes: Node[]): Map<string, string> {
  const labels = nodes.map(node => String(node.data.label || node.data.title
    || registry.get(String(node.data.definitionType))?.label || 'Node'));
  const counts = new Map<string, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  const ordinals = new Map<string, number>();
  const names = new Map<string, string>();
  const used = new Set<string>();
  nodes.forEach((node, index) => {
    const label = labels[index]!;
    let name = label;
    if (counts.get(label)! > 1) {
      let ordinal = (ordinals.get(label) ?? 0) + 1;
      while (counts.has(`${label} (${ordinal})`) || used.has(`${label} (${ordinal})`)) ordinal++;
      name = `${label} (${ordinal})`;
      ordinals.set(label, ordinal);
    }
    used.add(name);
    names.set(node.id, name);
  });
  return names;
}
