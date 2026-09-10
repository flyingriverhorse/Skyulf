import { useNodeInspection } from '../../core/hooks/useNodeInspection';
import { useInspectionSelection } from './nodeInspection/useInspectionSelection';
import { MeasuredTable } from './nodeInspection/MeasuredTable';
import { MeasuredChanges } from './nodeInspection/MeasuredChanges';
import { InspectionStatus } from './nodeInspection/InspectionStatus';
import { InspectionSelectors } from './nodeInspection/InspectionSelectors';
import { PredictedSchemaTable } from './nodeInspection/PredictedSchemaTable';

/** Inspect a selected node's captured input or output from data preview. */
export function NodeInspectionPanel({ nodeId, side }: { nodeId: string; side: 'input' | 'output' }) {
  const { branches, runId, isStale, isLoading, error, predictedSchema, blockReason } = useNodeInspection(nodeId);
  const selection = useInspectionSelection(nodeId, runId, branches, side);
  const { branch, capturedSide, table } = selection;

  return <div className="min-w-0 space-y-4">
    <div className="space-y-2">
      {blockReason && <p className="text-xs text-muted-foreground">{blockReason}</p>}
      <p className="text-xs leading-relaxed text-muted-foreground">
        Preview uses up to 1,000 source rows. Counts describe data at this node within that preview.
      </p>
    </div>

    <InspectionStatus isLoading={isLoading} isStale={isStale} capturedSide={capturedSide} side={side} error={error} />
    <InspectionSelectors nodeId={nodeId} runId={runId} branches={branches} side={side} selection={selection} />
    {table ? <>
      <MeasuredTable table={table} side={side} />
      {branch && side === 'output' && <MeasuredChanges branch={branch} output={table} />}
    </> : side === 'output' && predictedSchema && <PredictedSchemaTable predicted={predictedSchema} />}
  </div>;
}
