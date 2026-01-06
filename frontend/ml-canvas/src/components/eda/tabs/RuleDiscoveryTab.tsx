import React from 'react';
import { RuleTreeGraph } from '../RuleTreeGraph';
import { AlertCircle, GitBranch, BarChart3 } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../../ui/tooltip';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from 'recharts';

interface RuleDiscoveryTabProps {
    profile: any;
}

export const RuleDiscoveryTab: React.FC<RuleDiscoveryTabProps> = ({ profile }) => {
    const ruleTree = profile?.rule_tree;

    if (!ruleTree) {
        return (
            <div className="flex flex-col items-center justify-center h-64 text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8">
                <GitBranch className="w-12 h-12 mb-4 text-gray-300 dark:text-gray-600" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">Decision Tree Not Available</h3>
                <p className="text-center max-w-md mb-4">
                    No tree was generated. This usually happens if the dataset is too small, 
                    or if a target column was not selected for analysis.
                </p>
                <div className="flex items-center gap-2 text-sm text-amber-600 bg-amber-50 dark:bg-amber-900/20 dark:text-amber-400 px-4 py-2 rounded-md">
                    <AlertCircle className="w-4 h-4" />
                    <span>Ensure you have selected a <strong>Target Column</strong> and re-run the analysis.</span>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                            <GitBranch className="w-5 h-5 text-blue-600" />
                            Decision Tree Analysis
                        </h2>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                            This is a <strong>surrogate model</strong>: a simple decision tree trained on your dataset to approximate the relationship between features and the target.
                            Use it for interpretation (human-readable rules), not as a causal claim or a production model.
                        </p>
                    </div>
                    {ruleTree.accuracy && (
                        <TooltipProvider>
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <div className="text-sm bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 px-3 py-1 rounded-full border border-blue-200 dark:border-blue-800 cursor-help">
                                        Tree Fidelity: <strong>{(ruleTree.accuracy * 100).toFixed(1)}%</strong>
                                    </div>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p className="max-w-xs">
                                        <strong>Tree Fidelity:</strong> How closely this surrogate tree matches the patterns in your data (higher is better). If fidelity is low, treat extracted rules as exploratory.
                                    </p>
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    )}
                </div>

                <RuleTreeGraph tree={ruleTree} />
            </div>

            {ruleTree.feature_importances && ruleTree.feature_importances.length > 0 && (
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-md font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                        <BarChart3 className="w-5 h-5 text-blue-600" />
                        Feature Importance (Surrogate Model)
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                        Importances reflect what the <strong>surrogate tree</strong> used to split the data. They may differ from a more complex model’s importances.
                    </p>
                    <div className="h-80 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                                data={ruleTree.feature_importances.slice(0, 10)} // Top 10 features
                                layout="vertical"
                                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e5e7eb" />
                                <XAxis type="number" domain={[0, 'auto']} hide />
                                <YAxis 
                                    dataKey="feature" 
                                    type="category" 
                                    width={150} 
                                    tick={{ fontSize: 12, fill: '#6b7280' }}
                                    interval={0}
                                />
                                <RechartsTooltip 
                                    formatter={(value: number) => [(value * 100).toFixed(1) + '%', 'Importance']}
                                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '8px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                    cursor={{ fill: 'transparent' }}
                                />
                                <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={20}>
                                    {ruleTree.feature_importances.slice(0, 10).map((_: any, index: number) => (
                                        <Cell key={`cell-${index}`} fill={`hsl(217, 91%, ${60 - index * 3}%)`} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
                        Showing top 10 features driving the decision rules.
                    </p>
                </div>
            )}

            {ruleTree.rules && ruleTree.rules.length > 0 && (
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-md font-semibold text-gray-900 dark:text-white mb-4">Segmentation-Decision Tree (Surrogate Model)</h3>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                        {ruleTree.rules.map((rule: string, idx: number) => (
                            <div key={idx} className="p-3 bg-gray-50 dark:bg-gray-900 rounded border border-gray-100 dark:border-gray-700 text-sm font-mono text-gray-700 dark:text-gray-300">
                                {rule}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};
