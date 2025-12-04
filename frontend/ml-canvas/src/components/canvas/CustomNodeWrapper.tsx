import { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { registry } from '../../core/registry/NodeRegistry';
import { AlertCircle } from 'lucide-react';

export const CustomNodeWrapper = memo(({ data, selected }: NodeProps) => {
  const definitionType = data.definitionType as string;
  const definition = registry.get(definitionType);

  if (!definition) {
    return (
      <div className="p-4 border-2 border-destructive bg-destructive/10 rounded-md min-w-[150px]">
        <div className="flex items-center gap-2 text-destructive">
          <AlertCircle size={16} />
          <span className="text-sm font-bold">Unknown Node</span>
        </div>
        <div className="text-xs mt-1">Type: {definitionType}</div>
      </div>
    );
  }

  // Determine handle positions based on port definitions
  // This is a simplified version. In a real app, we might want more control over handle placement.
  
  return (
    <div className={`
      min-w-[200px] bg-card border-2 rounded-lg shadow-sm transition-all
      ${selected ? 'border-primary ring-2 ring-primary/20' : 'border-border hover:border-primary/50'}
    `}>
      {/* Header */}
      <div className="flex items-center p-3 border-b bg-muted/30 rounded-t-lg">
        <div className="p-1.5 bg-primary/10 rounded mr-3">
          {definition.icon && <definition.icon className="w-4 h-4 text-primary" />}
        </div>
        <div>
          <div className="text-sm font-bold">{definition.label}</div>
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
            {definition.category}
          </div>
        </div>
      </div>

      {/* Body (Custom Component or Default) */}
      <div className="p-3">
        {definition.component ? (
          <definition.component data={data} />
        ) : (
          <div className="text-xs text-muted-foreground italic">
            No configuration needed
          </div>
        )}
      </div>

      {/* Input Handles */}
      {definition.inputs.map((input, index) => (
        <Handle
          key={`input-${input.id}`}
          type="target"
          position={Position.Left}
          id={input.id}
          className="!w-3 !h-3 !bg-muted-foreground hover:!bg-primary transition-colors"
          style={{ top: `${((index + 1) * 100) / (definition.inputs.length + 1)}%` }}
        >
          <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground pointer-events-none whitespace-nowrap">
            {input.label}
          </div>
        </Handle>
      ))}

      {/* Output Handles */}
      {definition.outputs.map((output, index) => (
        <Handle
          key={`output-${output.id}`}
          type="source"
          position={Position.Right}
          id={output.id}
          className="!w-3 !h-3 !bg-muted-foreground hover:!bg-primary transition-colors"
          style={{ top: `${((index + 1) * 100) / (definition.outputs.length + 1)}%` }}
        >
          <div className="absolute right-4 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground pointer-events-none whitespace-nowrap">
            {output.label}
          </div>
        </Handle>
      ))}
    </div>
  );
});
