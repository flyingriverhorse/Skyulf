import React, { useCallback } from 'react';
import { CausalGraph } from '../CausalGraph';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { Network, Download } from 'lucide-react';
import { toPng } from 'html-to-image';

interface CausalTabProps {
    profile: any;
}

export const CausalTab: React.FC<CausalTabProps> = ({ profile }) => {
    const graph = profile.causal_graph;

    const handleDownload = useCallback(() => {
        const el = document.querySelector('#causal-chart .react-flow__viewport') as HTMLElement;
        if (!el) return;

        const isDark = document.documentElement.classList.contains('dark');
        const bgColor = isDark ? '#111827' : '#f9fafb';

        toPng(el, {
            backgroundColor: bgColor,
            pixelRatio: 2,
            filter: (node) => {
                // Exclude minimap / controls from export
                const cls = node?.classList;
                if (!cls) return true;
                return !cls.contains('react-flow__controls') && !cls.contains('react-flow__minimap');
            },
        })
            .then((dataUrl) => {
                const a = document.createElement('a');
                a.download = 'causal-discovery.png';
                a.href = dataUrl;
                a.click();
            })
            .catch((err) => console.error('Causal graph download failed', err));
    }, []);

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
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                        <Network className="w-5 h-5 mr-2 text-purple-500" />
                        Causal Discovery (PC Algorithm)
                        <InfoTooltip text="Inferred causal structure using the Peter-Clark algorithm. Arrows (A->B) suggest A causes B. Lines (A-B) suggest correlation without clear direction. Bidirected arrows (A<->B) suggest a hidden common cause." />
                    </h3>
                    <button
                        onClick={handleDownload}
                        className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                        title="Download Chart"
                    >
                        <Download className="w-4 h-4" />
                    </button>
                </div>
                <div className="mb-4 text-sm text-gray-600 dark:text-gray-400 bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded border border-yellow-200 dark:border-yellow-800">
                    <strong>Disclaimer:</strong> This graph is inferred purely from observational data. Correlation does not imply causation. Use this as a hypothesis generation tool, not absolute truth.
                </div>
                <div id="causal-chart">
                    <CausalGraph graph={graph} />
                </div>
            </div>
        </div>
    );
};
