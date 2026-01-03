import React from 'react';
import { Lightbulb, X, CheckCircle, RefreshCw } from 'lucide-react';

interface InsightsTabProps {
    profile: any;
}

export const InsightsTab: React.FC<InsightsTabProps> = ({ profile }) => {
    return (
        <div className="mt-4 space-y-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                    <Lightbulb className="w-5 h-5 mr-2 text-yellow-500" />
                    Actionable Recommendations
                </h3>
                {profile.recommendations && profile.recommendations.length > 0 ? (
                    <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                        {profile.recommendations.map((rec: any, idx: number) => (
                            <div key={idx} className="flex items-start p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-100 dark:border-gray-800">
                                <div className={`p-2 rounded-full mr-4 shrink-0 ${
                                    rec.action === 'Drop' ? 'bg-red-100 text-red-600' :
                                    rec.action === 'Impute' ? 'bg-blue-100 text-blue-600' :
                                    rec.action === 'Transform' ? 'bg-purple-100 text-purple-600' :
                                    rec.action === 'Resample' ? 'bg-orange-100 text-orange-600' :
                                    'bg-green-100 text-green-600'
                                }`}>
                                    {rec.action === 'Drop' ? <X className="w-4 h-4" /> : 
                                        rec.action === 'Keep' ? <CheckCircle className="w-4 h-4" /> :
                                        <RefreshCw className="w-4 h-4" />}
                                </div>
                                <div>
                                    <h4 className="text-sm font-semibold text-gray-900 dark:text-white">
                                        {rec.action} {rec.column && <span className="text-gray-500">'{rec.column}'</span>}
                                    </h4>
                                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{rec.suggestion}</p>
                                    <p className="text-xs text-gray-400 mt-2 italic">Reason: {rec.reason}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-gray-500 italic">No specific recommendations found. Your data looks clean!</p>
                )}
            </div>
        </div>
    );
};
