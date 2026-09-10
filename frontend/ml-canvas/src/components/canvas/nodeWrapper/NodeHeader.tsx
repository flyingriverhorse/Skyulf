import type { NodeDefinition } from '../../../core/types/nodes';
import type { NodePresentation } from './useNodePresentation';
import { Merge, GitFork } from 'lucide-react';

function MergeBadge({ merge }: Pick<NodePresentation, 'merge'>) {
  const { showMergeBadge, isParallel, mergeWarningSeverity, incomingSourceCount } = merge;
  return (<>
    {showMergeBadge && (
      <span
        className={`shrink-0 flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded-full font-medium ${isParallel
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
  </>);
}

export function NodeHeader({ definition, merge, schema }: Pick<NodePresentation, 'merge' | 'schema'>
  & { definition: NodeDefinition<unknown> }) {
  const { predictedSchema, schemaIsDataDependent } = schema;
  return (
    <div className="flex items-center p-3 border-b bg-muted/30 rounded-t-lg">
      <div className="p-1.5 bg-primary/10 rounded mr-3 shrink-0">
        {definition.icon && <definition.icon className="w-4 h-4 text-primary" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <div className="text-sm font-bold truncate" title={definition.label}>{definition.label}</div>
          <MergeBadge merge={merge} />
          {predictedSchema && (
            <span
              className="shrink-0 flex items-center text-[10px] px-1.5 py-0.5 rounded-full font-medium bg-muted/60 text-muted-foreground border border-border/60"
              title={
                `Canvas schema preview — predicted output before running:\n` +
                `${predictedSchema.columns.length} column${predictedSchema.columns.length === 1 ? '' : 's'}:\n` +
                predictedSchema.columns.slice(0, 12).join(', ') +
                (predictedSchema.columns.length > 12 ? `, … (+${predictedSchema.columns.length - 12} more)` : '')
              }
            >
              ↳ {predictedSchema.columns.length} col{predictedSchema.columns.length === 1 ? '' : 's'}
            </span>
          )}
          {schemaIsDataDependent && (
            <span
              className="shrink-0 flex items-center text-[10px] px-1.5 py-0.5 rounded-full font-medium bg-muted/30 text-muted-foreground/60 border border-dashed border-border/50"
              title="Schema depends on data — this step's output columns can only be known after running the pipeline (e.g. one-hot encoding adds one column per category). Everything is fine; run the pipeline to see the actual output."
            >
              ↳ ?
            </span>
          )}
        </div>
        <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
          {definition.category}
        </div>
      </div>
    </div>
  );
}
