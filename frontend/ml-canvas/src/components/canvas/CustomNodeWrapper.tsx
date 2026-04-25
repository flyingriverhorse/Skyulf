import { memo } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from '@xyflow/react';
import { registry } from '../../core/registry/NodeRegistry';
import { AlertCircle, X, CheckCircle2, XCircle, Merge, GitFork } from 'lucide-react';
import { useGraphStore } from '../../core/store/useGraphStore';
import type { NodeExecutionResult } from '../../core/api/client';

const TRAINING_TYPES = new Set(['basic_training', 'advanced_tuning']);
// Terminals that auto-split each upstream input into its own parallel
// branch / preview tab instead of column-merging them. Mirror
// AUTO_PARALLEL_STEP_TYPES in backend graph_utils.py.
const AUTO_PARALLEL_TYPES = new Set(['data_preview']);

function CustomNodeWrapperImpl({ id, data, selected }: NodeProps) {
  const definitionType = data.definitionType as string;
  const definition = registry.get(definitionType);
  const { deleteElements } = useReactFlow();
  
  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult: NodeExecutionResult | undefined = executionResult?.node_results?.[id];
  const incomingSourceCount = useGraphStore(
    (state) => new Set(state.edges.filter((e) => e.target === id).map((e) => e.source)).size,
    (a, b) => a === b
  );

  // Has this node received an active sibling-fan-in advisory with real
  // overlap (i.e. last-wins overwrite is happening)? If so, color the
  // merge badge amber so the canvas itself flags the risk.
  const mergeWarningSeverity: 'risk' | 'safe' | null = (() => {
    const warnings = executionResult?.merge_warnings ?? [];
    const w = warnings.find((mw) => mw.node_id === id);
    if (!w) return null;
    return (w.overlap_columns?.length ?? 0) > 0 ? 'risk' : 'safe';
  })();

  // Parallel badge: training nodes when user explicitly chose parallel mode,
  // OR auto-parallel terminals (data_preview) wired to 2+ sources.
  const isTrainingNode = TRAINING_TYPES.has(definitionType);
  const isAutoParallel = AUTO_PARALLEL_TYPES.has(definitionType);
  const isParallel =
    (isTrainingNode && data.execution_mode === 'parallel') ||
    (isAutoParallel && incomingSourceCount > 1);

  // Only nodes that actually consume upstream data (i.e. declare an input
  // port) can merge. Dataset/data-loader nodes have no inputs and must not
  // show a merge badge even if React Flow allowed an edge in. Auto-parallel
  // terminals show the parallel badge instead of the merge badge.
  const canMerge = (definition?.inputs?.length ?? 0) > 0 && !isAutoParallel;
  const showMergeBadge = (canMerge && incomingSourceCount > 1) || isParallel;

  const onDelete = (evt: React.MouseEvent) => {
    evt.stopPropagation();
    deleteElements({ nodes: [{ id }] });
  };

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
      min-w-[200px] bg-card border-2 rounded-lg shadow-sm transition-all duration-150
      ${selected
        ? 'border-primary shadow-lg shadow-primary/30 scale-[1.02]'
        : 'border-border hover:border-primary/50'}
    `}>
      {/* Header */}
      <div className="flex items-center p-3 border-b bg-muted/30 rounded-t-lg relative group">
        <div className="p-1.5 bg-primary/10 rounded mr-3">
          {definition.icon && <definition.icon className="w-4 h-4 text-primary" />}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <div className="text-sm font-bold">{definition.label}</div>
            {showMergeBadge && (
              <span
                className={`flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
                  isParallel
                    ? 'bg-amber-500/15 text-amber-500'
                    : mergeWarningSeverity === 'risk'
                    ? 'bg-amber-500/20 text-amber-600 dark:text-amber-400 ring-1 ring-amber-500/40'
                    : 'bg-blue-500/15 text-blue-400'
                }`}
                title={
                  isParallel
                    ? `Parallel: ${incomingSourceCount} branches will run as separate experiments`
                    : mergeWarningSeverity === 'risk'
                    ? `Merge with overlap: ${incomingSourceCount} branches share columns — last input overwrites earlier ones. See Results panel banner for details.`
                    : `Merge: combining data from ${incomingSourceCount} upstream sources`
                }
              >
                {isParallel ? <GitFork size={10} /> : <Merge size={10} />}
                {incomingSourceCount}
              </span>
            )}
            {nodeResult && (
              <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium border ${
                nodeResult.status === 'success' 
                  ? 'bg-green-50 text-green-700 border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-900' 
                  : 'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-900'
              }`}>
                {nodeResult.status === 'success' ? <CheckCircle2 size={10} /> : <XCircle size={10} />}
              </div>
            )}
          </div>
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
            {definition.category}
          </div>
        </div>

        {/* Delete Button */}
        <button
          onClick={onDelete}
          className="p-1 rounded-md hover:bg-destructive/10 hover:text-destructive text-muted-foreground/50 transition-all opacity-0 group-hover:opacity-100 ml-2"
          title="Remove Node"
        >
          <X size={14} />
        </button>
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
}

export const CustomNodeWrapper = memo(CustomNodeWrapperImpl);
