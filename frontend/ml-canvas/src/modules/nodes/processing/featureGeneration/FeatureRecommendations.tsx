import { useState } from 'react';
import { ChevronDown, ChevronRight, Lightbulb } from 'lucide-react';
import { useRecommendations } from '../../../../core/hooks/useRecommendations';
import { RecommendationsPanel } from '../../../../components/panels/RecommendationsPanel';

export function FeatureRecommendations({ nodeId }: { nodeId: string | undefined }) {
  const [showRecommendations, setShowRecommendations] = useState(false);
  // Recommendations
  const recommendations = useRecommendations(nodeId || '', {
    types: ['feature_generation'],
    suggestedNodeTypes: ['FeatureGenerationNode'],
    scope: 'column'
  });

  return <>
    {recommendations.length > 0 && (
      <div className="border-b bg-muted/10">
        <button
          onClick={() => { setShowRecommendations(!showRecommendations); }}
          className="w-full flex items-center justify-between px-4 py-2 text-xs font-medium text-muted-foreground hover:text-primary hover:bg-muted/20 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Lightbulb size={14} className={recommendations.length > 0 ? "text-yellow-500" : ""} />
            <span>Recommendations ({recommendations.length})</span>
          </div>
          {showRecommendations ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>

        {showRecommendations && (
          <div className="p-4 bg-muted/5 border-t">
            <RecommendationsPanel
              recommendations={recommendations}
              className="mb-0"
            />
          </div>
        )}
      </div>
    )}
  </>;
}
