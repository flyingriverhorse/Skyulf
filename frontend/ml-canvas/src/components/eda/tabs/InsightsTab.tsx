import React from 'react';
import { Lightbulb } from 'lucide-react';

interface InsightsTabProps {
    profile: any;
}

export const InsightsTab: React.FC<InsightsTabProps> = ({ profile }) => {
    return (
        <div className="mt-4 space-y-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                    <Lightbulb className="w-5 h-5 mr-2 text-yellow-500" />
                    Smart Recommendations
                </h3>
                {profile.recommendations && profile.recommendations.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {profile.recommendations.map((rec: any, idx: number) => (
                            <div key={idx} className="p-4 bg-blue-50 dark:bg-blue-900/10 rounded-lg border border-blue-100 dark:border-blue-800">
                                <div className="flex justify-between items-start mb-2">
                                    <span className="text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400 bg-white dark:bg-gray-800 px-2 py-1 rounded border border-blue-100 dark:border-blue-800">
                                        {rec.action}
                                    </span>
                                    {rec.column && (
                                        <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                                            {rec.column}
                                        </span>
                                    )}
                                </div>
                                <h4 className="font-medium text-gray-900 dark:text-white text-sm mb-1">{rec.reason}</h4>
                                <p className="text-sm text-gray-600 dark:text-gray-300">{rec.suggestion}</p>
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
