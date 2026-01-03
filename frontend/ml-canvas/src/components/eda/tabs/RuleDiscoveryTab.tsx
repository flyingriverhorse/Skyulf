import React from 'react';
import { RuleTreeGraph } from '../RuleTreeGraph';
import { AlertCircle, GitBranch } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../../ui/tooltip';

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
                            Visualize how different features combine to predict the target. 
                            This tree helps identify key segments and decision rules in your data.
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
                                        <strong>Tree Fidelity (Accuracy):</strong> Indicates how well this simplified tree mimics the patterns in your dataset. A higher percentage means these rules are more reliable.
                                    </p>
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    )}
                </div>

                <RuleTreeGraph tree={ruleTree} />
            </div>
        </div>
    );
};
