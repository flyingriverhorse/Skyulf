import type { GraphNode } from './types';

function friendlyStep(stepType: string): string {
  if (!stepType) return '';
  if (stepType.includes('_')) {
    return stepType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  }
  return stepType.replace(/(?<!^)(?=[A-Z])/g, ' ').trim();
}

/** Keep real column names while bounding long array summaries. */
function renderStepValue(value: unknown): string {
  if (Array.isArray(value)) {
    const items = value.map(item => String(item));
    return items.length > 4
      ? `[${items.slice(0, 4).join(', ')}, +${items.length - 4} more]`
      : `[${items.join(', ')}]`;
  }
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

const DETAIL_KEYS = ['method', 'strategy', 'columns', 'target_column', 'test_size', 'val_size', 'random_state', 'n_neighbors'];

function stepDetails(node: GraphNode): string[] {
  const details: string[] = [];
  for (const key of DETAIL_KEYS) {
    const value = node.params?.[key];
    if (value === undefined || value === null || value === '') continue;
    if (Array.isArray(value) && value.length === 0) continue;
    details.push(`${key}=${renderStepValue(value)}`);
  }
  return details;
}

/** Combine primary and secondary operands and cap the displayed argument list. */
function operationArguments(operation: Record<string, unknown>): string {
  const inputs = Array.isArray(operation['input_columns']) ? operation['input_columns'] as unknown[] : [];
  const secondary = Array.isArray(operation['secondary_columns']) ? operation['secondary_columns'] as unknown[] : [];
  const operands = [...inputs, ...secondary].map(String);
  return operands.length > 2
    ? `${operands.slice(0, 2).join(', ')}, +${operands.length - 2}`
    : operands.join(', ');
}

/** Feature-generation operations use their own method and input columns. */
function summarizeOperation(raw: unknown): string {
  const operation = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
  const method = String(operation['method'] ?? operation['operation_type'] ?? 'op');
  const args = operationArguments(operation);
  return args ? `${method}(${args})` : method;
}

export function summarizeStep(node: GraphNode): string {
  const display = (node.params?._display_name as string | undefined) || friendlyStep(String(node.step_type || ''));
  const detail = stepDetails(node);
  const operations = node.params?.['operations'];
  if (Array.isArray(operations) && operations.length > 0) {
    const formatted = operations.slice(0, 3).map(summarizeOperation);
    const tail = operations.length > 3 ? `, +${operations.length - 3} more` : '';
    detail.push(`ops=[${formatted.join(', ')}${tail}]`);
  }
  return detail.length > 0 ? `${display} (${detail.join(', ')})` : display;
}
