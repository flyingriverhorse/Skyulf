import React from 'react';
import { CausalGraph } from '../CausalGraph';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { Network } from 'lucide-react';

interface CausalTabProps {
    profile: any;
}

export const CausalTab: React.FC<CausalTabProps> = ({ profile }) => {
    const graph = profile.causal_graph;

    if (!graph) {
        return (
            <div className="p-8 text-center text-gray-500">
                <Network className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No causal graph available. This might be because there are fewer than 2 numeric columns or the analysis failed.</p>
            </div>
        );
    }

    return (
        <div className="space-y-6 mt-4">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                    <Network className="w-5 h-5 mr-2 text-purple-500" />
                    Causal Discovery (PC Algorithm)
                    <InfoTooltip text="Inferred causal structure using the Peter-Clark algorithm. Arrows (A->B) suggest A causes B. Lines (A-B) suggest correlation without clear direction. Bidirected arrows (A<->B) suggest a hidden common cause." />
                </h3>
                <div className="mb-4 text-sm text-gray-600 dark:text-gray-400 bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded border border-yellow-200 dark:border-yellow-800">
                    <strong>Disclaimer:</strong> This graph is inferred purely from observational data. Correlation does not imply causation. Use this as a hypothesis generation tool, not absolute truth.
                </div>
                <CausalGraph graph={graph} />
            </div>
        </div>
    );
};
