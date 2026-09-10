import { Calculator } from 'lucide-react';
import { useIsWideContainer } from '../../../../core/hooks/useIsWideContainer';
import type { NodeSettingsProps } from '../../../../core/types/nodes';
import type { FeatureGenerationConfig } from './types';
import { useFeatureColumns } from './useFeatureColumns';
import { useOperations } from './useOperations';
import { OperationToolbar } from './OperationToolbar';
import { OperationCard } from './OperationCard';
import { FeatureRecommendations } from './FeatureRecommendations';
import { FeatureExecutionFeedback } from './FeatureExecutionFeedback';

export function FeatureGenerationSettings({ config, onChange, nodeId }: NodeSettingsProps<FeatureGenerationConfig>) {
  const columns = useFeatureColumns(nodeId);
  const operations = useOperations(config, onChange);
  // Responsive layout: switch to a 2-column layout once the panel is wider than 380px.
  const [containerRef, isWide] = useIsWideContainer(380);
  return (
    <div ref={containerRef} className={isWide ? 'flex min-h-[300px]' : 'flex flex-col min-h-[300px]'}>
      <OperationToolbar isWide={isWide} addOperation={operations.addOperation} />
      <div className="flex-1 flex flex-col min-w-0 bg-background">
        <FeatureRecommendations nodeId={nodeId} />
        <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-[240px]">
          {(config.operations || []).map((op, idx) => (
            <OperationCard key={idx} op={op} idx={idx} {...columns} {...operations} />
          ))}
          {(!config.operations || config.operations.length === 0) && (
            <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed rounded-lg bg-muted/5 text-muted-foreground">
              <Calculator size={32} className="mb-2 opacity-20" />
              <p className="text-xs font-medium">No operations added</p>
              <p className="text-[10px] opacity-70">Select a type above to start</p>
            </div>
          )}
          <FeatureExecutionFeedback nodeId={nodeId} />
        </div>
      </div>
    </div>
  );
}
