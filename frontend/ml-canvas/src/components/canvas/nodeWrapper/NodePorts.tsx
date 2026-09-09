import type { NodeDefinition } from '../../../core/types/nodes';
import { ConnectionPort } from '../ConnectionPort';

interface PortProps {
  id: string;
  definition: NodeDefinition<unknown>;
  canConnect: boolean;
}

export function NodeOutputPorts({ id, definition, canConnect, definitionType, data }: PortProps & {
  definitionType: string; data: Record<string, unknown>;
}) {
  const hasMultipleOutputs = definition.outputs.length > 1;
  return definition.outputs.map((output, index) => (
    <ConnectionPort
      key={`output-${output.id}`}
      nodeId={id} direction="source" port={output}
      canConnect={canConnect}
      disabled={definitionType === 'TrainTestSplitter' && output.id === 'validation' && !(Number(data.validation_size ?? 0) > 0)}
      compact={hasMultipleOutputs}
      top={hasMultipleOutputs ? `calc(50% + ${(index - (definition.outputs.length - 1) / 2) * 16 - 12}px)` : '50%'}
    />
  ));
}

export function NodeInputPorts({ id, definition, canConnect }: PortProps) {
  return definition.inputs.map((input, index) => (
    <ConnectionPort
      key={`input-${input.id}`}
      nodeId={id} direction="target" port={input}
      canConnect={canConnect}
      top={`${((index + 1) * 100) / (definition.inputs.length + 1)}%`}
    />
  ));
}
