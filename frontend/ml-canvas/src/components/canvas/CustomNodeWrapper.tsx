import { memo } from 'react';
import { NodeProps, useReactFlow } from '@xyflow/react';
import { registry } from '../../core/registry/NodeRegistry';
import { useNodePresentation } from './nodeWrapper/useNodePresentation';
import { useSplitOutputSpace } from './nodeWrapper/useSplitOutputSpace';
import { nodeCardClass, nodeFeedbackAttributes } from './nodeWrapper/nodeCardClass';
import { NodeBody } from './nodeWrapper/NodeBody';
import { NodeHeader } from './nodeWrapper/NodeHeader';
import { NodeInputPorts, NodeOutputPorts } from './nodeWrapper/NodePorts';
import { UnknownNode, NodeDeleteButton, NodeStatus, NodePerformanceFooter } from './nodeWrapper/NodeStatus';

function CustomNodeWrapperImpl({ id, data, selected, isConnectable }: NodeProps) {
  const definitionType = data.definitionType as string;
  const definition = registry.get(definitionType);
  const splitBodyRef = useSplitOutputSpace(definition);
  const { deleteElements, getEdges } = useReactFlow();
  const presentation = useNodePresentation(id, data, definitionType, definition);
  const { readOnly, execution, schema, leakage, merge, validation, perf } = presentation;
  const onDelete = (event: React.MouseEvent) => {
    event.stopPropagation();
    deleteElements({ nodes: [{ id }] });
  };

  if (!definition) return <UnknownNode definitionType={definitionType} />;

  const hasMultipleOutputs = definition.outputs.length > 1;
  const canConnect = isConnectable !== false && !readOnly;
  const outputPorts = <NodeOutputPorts id={id} definition={definition} definitionType={definitionType}
    data={data} canConnect={canConnect} />;
  return (
    <div
      data-testid={`canvas-node-${definitionType}`}
      data-node-definition-type={definitionType}
      {...nodeFeedbackAttributes(presentation)}
      className={nodeCardClass({ ...presentation, selected })}
    >
      <NodeDeleteButton readOnly={readOnly} selected={selected} onDelete={onDelete} />
      <NodeStatus definition={definition} leakage={leakage} execution={execution} schema={schema} validation={validation} />
      <NodeHeader definition={definition} merge={merge} schema={schema} />
      <div ref={splitBodyRef} className={hasMultipleOutputs ? 'relative' : undefined}>
        <NodeBody id={id} data={data} definition={definition} execution={execution}
          hasMultipleOutputs={hasMultipleOutputs} getEdges={getEdges} />
        {hasMultipleOutputs && outputPorts}
      </div>
      <NodePerformanceFooter perf={perf} />
      <NodeInputPorts id={id} definition={definition} canConnect={canConnect} />
      {!hasMultipleOutputs && outputPorts}
    </div>
  );
}

export const CustomNodeWrapper = memo(CustomNodeWrapperImpl);
