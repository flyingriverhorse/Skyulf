import React from 'react';
import { ScatterChart as ScatterIcon, Download, Box, Info } from 'lucide-react';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { ThreeDScatterPlot } from '../ThreeDScatterPlot';
import { CanvasScatterPlot } from '../CanvasScatterPlot';

interface PCATabProps {
    profile: any;
    isPCA3D: boolean;
    setIsPCA3D: (value: boolean) => void;
    downloadChart: (elementId: string, filename: string, title?: string, subtitle?: string) => void;
}

export const PCATab: React.FC<PCATabProps> = ({ profile, isPCA3D, setIsPCA3D, downloadChart }) => {
    return (
        <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                    <ScatterIcon className="w-5 h-5 mr-2 text-purple-500" />
                    Multivariate Structure (PCA)
                    <InfoTooltip text="Reduces data to 2D or 3D. Points closer together are similar. Colors show clusters or target values." />
                </h3>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => downloadChart('pca-chart', 'pca-analysis', 'Multivariate Structure (PCA)', isPCA3D ? '3D Projection' : '2D Projection')}
                        className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                        title="Download Chart"
                    >
                        <Download className="w-4 h-4" />
                    </button>
                    <button
                        onClick={() => setIsPCA3D(!isPCA3D)}
                        className={`p-2 rounded-md border transition-colors flex items-center gap-2 text-sm ${
                            isPCA3D 
                                ? 'bg-blue-50 border-blue-200 text-blue-700 dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-300' 
                                : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300'
                        }`}
                    >
                        <Box className="w-4 h-4" />
                        {isPCA3D ? 'Switch to 2D' : 'Switch to 3D'}
                    </button>
                </div>
            </div>

            <div id="pca-chart">
            {profile.pca_data ? (
                isPCA3D ? (
                    <ThreeDScatterPlot 
                        data={profile.pca_data} 
                        xKey="x" 
                        yKey="y" 
                        zKey="z"
                        labelKey="label"
                        xLabel="PC1"
                        yLabel="PC2"
                        zLabel="PC3"
                    />
                ) : (
                    <CanvasScatterPlot 
                        data={profile.pca_data} 
                        xKey="x" 
                        yKey="y" 
                        labelKey="label"
                        xLabel="Principal Component 1 (PC1)"
                        yLabel="Principal Component 2 (PC2)"
                    />
                )
            ) : (
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-900 rounded text-gray-400 text-sm text-center p-4">
                    Not enough numeric data for PCA.
                </div>
            )}
            </div>

            <div className="mt-6 flex items-start gap-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md border border-blue-100 dark:border-blue-800">
                <Info className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
                <div className="text-sm text-blue-700 dark:text-blue-300">
                    <p className="font-medium mb-1">About this visualization</p>
                    <p>
                        This chart shows a {isPCA3D ? '3D' : '2D'} projection of your data using Principal Component Analysis (PCA). 
                        Points that are close together share similar characteristics across all numeric features.
                    </p>
                    <p className="mt-1 text-xs opacity-80">
                        * For performance, this visualization uses a random sample of up to <strong>5,000 points</strong>.
                    </p>
                </div>
            </div>
        </div>
    );
};
