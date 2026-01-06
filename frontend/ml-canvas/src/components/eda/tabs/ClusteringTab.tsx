import React from 'react';
import { Network, Download, Info } from 'lucide-react';
import { InfoTooltip } from '../../ui/InfoTooltip';

interface ClusterStats {
    cluster_id: number;
    size: number;
    percentage: number;
    center: Record<string, number>;
}

interface ClusteringAnalysis {
    method: string;
    n_clusters: number;
    inertia: number;
    clusters: ClusterStats[];
    points: Array<{ x: number, y: number, cluster: number, label?: string }>;
}

interface ClusteringTabProps {
    profile: any;
    downloadChart: (elementId: string, filename: string, title?: string, subtitle?: string) => void;
}

export const ClusteringTab: React.FC<ClusteringTabProps> = ({ profile, downloadChart }) => {
    const analysis = profile.clustering as ClusteringAnalysis | undefined;

    if (!analysis) {
        return (
             <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm flex flex-col items-center justify-center text-gray-500 min-h-[300px]">
                <Network className="w-12 h-12 mb-4 opacity-20" />
                <p>Clustering analysis not available.</p>
                <p className="text-sm mt-2">Requires at least 2 numeric columns.</p>
            </div>
        );
    }

    return (
        <div className="mt-4 space-y-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                        <Network className="w-5 h-5 mr-2 text-rose-500" />
                        Post-Hoc Clustering (Segmentation)
                        <InfoTooltip text="Automatically groups data points into distinct clusters (segments) based on similarity. Useful for finding natural groups like 'High Spenders' or 'At Risk' users." />
                    </h3>
                    <button
                        onClick={() => downloadChart('clustering-chart', 'clustering-analysis', 'Clustering Segmentation', `${analysis.n_clusters} Clusters Found`)}
                        className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                        title="Download Chart"
                    >
                        <Download className="w-4 h-4" />
                    </button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Stats Section */}
                    <div className="lg:col-span-2 space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                             <div className="p-4 bg-purple-50 dark:bg-purple-900/10 rounded-lg border border-purple-100 dark:border-purple-800">
                                <h4 className="text-sm font-semibold text-purple-900 dark:text-purple-100 mb-2">Algorithm Summary</h4>
                                <div className="space-y-1 text-sm text-purple-700 dark:text-purple-300">
                                    <div className="flex justify-between">
                                        <span>Method:</span>
                                        <span className="font-medium">{analysis.method}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Metric (Inertia):</span>
                                        <span className="font-medium">{analysis.inertia.toFixed(1)}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Clusters:</span>
                                        <span className="font-medium">{analysis.n_clusters}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mt-4">Cluster Profiles (Centroids)</h4>
                        
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {analysis.clusters.map((cluster) => (
                                <div key={cluster.cluster_id} className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="font-bold text-gray-900 dark:text-white flex items-center">
                                            <span className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: getColorForIndex(cluster.cluster_id) }}></span>
                                            Cluster {cluster.cluster_id}
                                        </span>
                                        <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-gray-600 dark:text-gray-300">
                                            {cluster.percentage.toFixed(1)}% ({cluster.size})
                                        </span>
                                    </div>
                                    
                                    <div className="space-y-1">
                                        {Object.entries(cluster.center).slice(0, 5).map(([col, val]) => (
                                            <div key={col} className="flex justify-between text-xs">
                                                <span className="text-gray-500 dark:text-gray-400 truncate w-24" title={col}>{col}</span>
                                                <span className="font-mono text-gray-700 dark:text-gray-200">{val.toFixed(2)}</span>
                                            </div>
                                        ))}
                                        {Object.keys(cluster.center).length > 5 && (
                                            <div className="text-xs text-center text-gray-400 italic pt-1">
                                                + {Object.keys(cluster.center).length - 5} more features
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="flex items-start gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg text-sm border border-blue-100 dark:border-blue-800">
                <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <div>
                     These clusters are found using unsupervised learning (K-Means) on numeric features. 
                     The plot shows a 2D projection (PCA) of the clusters. Use this to identify distinct groups of data points that behave similarly.
                </div>
            </div>
        </div>
    );
};

// Helper for consistency with ChartJS colors
const getColorForIndex = (index: number) => {
    const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#FFBB28', '#FF8042'];
    return colors[index % colors.length];
};
