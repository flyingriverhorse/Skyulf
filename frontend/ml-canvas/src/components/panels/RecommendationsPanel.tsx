import React from 'react';
import { Recommendation } from '../../core/api/client';
import { Lightbulb, CheckCircle } from 'lucide-react';

interface RecommendationsPanelProps {
  recommendations: Recommendation[];
  onApply?: (rec: Recommendation) => void;
  className?: string;
}

export const RecommendationsPanel: React.FC<RecommendationsPanelProps> = ({
  recommendations,
  onApply,
  className = '',
}) => {
  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  return (
    <div className={`bg-blue-50 border border-blue-100 rounded-md p-3 ${className}`}>
      <div className="flex items-center gap-2 mb-2 text-blue-800 font-medium text-sm">
        <Lightbulb className="w-4 h-4" />
        <span>Recommendations</span>
      </div>
      <div className="space-y-2">
        {recommendations.map((rec) => (
          <div
            key={rec.rule_id}
            className="bg-white border border-blue-100 rounded p-2 text-sm shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="flex justify-between items-start gap-2">
              <div className="flex-1">
                <p className="text-gray-800 font-medium text-xs mb-1">
                  {rec.type.replace('_', ' ').toUpperCase()}
                </p>
                <p className="text-gray-600 text-xs leading-relaxed">
                  {rec.description}
                </p>
                {rec.reasoning && (
                  <p className="text-gray-400 text-[10px] mt-1 italic">
                    Why: {rec.reasoning}
                  </p>
                )}
              </div>
              {onApply && (
                <button
                  onClick={() => onApply(rec)}
                  className="text-blue-600 hover:text-blue-800 p-1 rounded hover:bg-blue-50"
                  title="Apply Recommendation"
                >
                  <CheckCircle className="w-5 h-5" />
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
