import type { Node } from '@xyflow/react';
import { registry } from '../../registry/NodeRegistry';
import type { BranchDef } from './types';

/** Generate distinct saturated colors with the existing golden-angle spacing. */
export function generateBranchColors(count: number): string[] {
  const colors: string[] = [];
  const goldenAngle = 137.508;
  for (let i = 0; i < count; i++) {
    const hue = (i * goldenAngle) % 360;
    colors.push(`hsl(${Math.round(hue)}, 80%, 65%)`);
  }
  return colors;
}

function prettifyModelType(modelType: string): string {
  return modelType.replace(/_classifier$|_regressor$/, '').split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

/** Distinguish repeated terminal model types without renumbering other groups. */
export function getTerminalSuffixes(terminals: Node[]): Map<string, string> {
  const byModel = new Map<string, string[]>();
  for (const terminal of terminals) {
    const type = (terminal.data.model_type as string | undefined)
      || (terminal.data.definitionType as string | undefined) || '';
    const ids = byModel.get(type) ?? [];
    ids.push(terminal.id);
    byModel.set(type, ids);
  }
  const suffixes = new Map<string, string>();
  for (const ids of byModel.values()) {
    if (ids.length < 2) continue;
    ids.forEach((id, index) => suffixes.set(id, `#${index + 1}`));
  }
  return suffixes;
}

function getSourceName(data: Record<string, unknown>): string {
  return (data.label as string) || (data.title as string)
    || (typeof data.definitionType === 'string'
      ? data.definitionType.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase()) : '');
}

function getTerminalName(data: Record<string, unknown>): string {
  return (data.label as string) || (data.title as string)
    || registry.get(String(data.definitionType))?.label
    || (typeof data.definitionType === 'string'
      ? data.definitionType.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/_/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase()) : '');
}

function getBranchName(branch: BranchDef, nodes: Node[]): string {
  const modelType = branch.terminal.data.model_type as string | undefined;
  let name = modelType ? prettifyModelType(modelType) : '';
  if (!name && branch.inputEdge) {
    const sourceId = branch.inputEdge.source;
    const source = nodes.find(node => node.id === sourceId);
    name = getSourceName(source?.data ?? {});
  }
  return name || getTerminalName(branch.terminal.data ?? {});
}

/** Use global path letters for multiple terminals and local letters otherwise. */
export function getBranchLabel(branch: BranchDef, index: number, terminals: Node[], nodes: Node[], suffixes: Map<string, string>): string {
  const letterIndex = terminals.length > 1 ? index : branch.localIndex;
  const letter = String.fromCharCode(65 + letterIndex);
  let name = getBranchName(branch, nodes);
  const suffix = suffixes.get(branch.terminal.id);
  if (suffix && name) name = `${name} ${suffix}`;
  return name ? `Path ${letter} · ${name}` : `Path ${letter}`;
}
