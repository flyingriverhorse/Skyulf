import type { Node } from '@xyflow/react';

/** Find the nearest clear slot to the right, searching above as well as below crowded nodes. */
export function nextNodePosition(source: Node, nodes: Node[]): { x: number; y: number } {
  const gap = 64;
  const width = 280;
  const height = 160;
  const origin = { x: source.position.x + (source.measured?.width ?? source.width ?? 240) + gap, y: source.position.y };
  const candidates = [origin];
  for (const node of nodes) {
    const right = node.position.x + (node.measured?.width ?? node.width ?? width) + gap;
    const above = node.position.y - height - gap;
    const below = node.position.y + (node.measured?.height ?? node.height ?? height) + gap;
    candidates.push({ x: origin.x, y: above }, { x: origin.x, y: below }, { x: Math.max(origin.x, right), y: origin.y });
  }
  const clear = (position: { x: number; y: number }) => nodes.every(node =>
    position.x + width + 24 <= node.position.x || position.x >= node.position.x + (node.measured?.width ?? node.width ?? width) + 24 ||
    position.y + height + 24 <= node.position.y || position.y >= node.position.y + (node.measured?.height ?? node.height ?? height) + 24);
  candidates.sort((a, b) => Math.hypot(a.x - origin.x, a.y - origin.y) - Math.hypot(b.x - origin.x, b.y - origin.y));
  // The candidate beyond the rightmost node always provides a finite fallback.
  return candidates.find(clear) ?? origin;
}
