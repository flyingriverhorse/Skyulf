import React from 'react';
import { 
    Download, 
    ScatterChart as ScatterIcon, 
    Box 
} from 'lucide-react';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { ThreeDScatterPlot } from '../ThreeDScatterPlot';
import { CanvasScatterPlot } from '../CanvasScatterPlot';

interface BivariateTabProps {
    profile: any;
    downloadChart: (id: string, filename: string, title: string, subtitle?: string) => void;
    scatterX: string;
    setScatterX: (value: string) => void;
    scatterY: string;
    setScatterY: (value: string) => void;
    scatterZ: string;
    setScatterZ: (value: string) => void;
    scatterColor: string;
    setScatterColor: (value: string) => void;
    is3D: boolean;
    setIs3D: (value: boolean) => void;
}

export const BivariateTab: React.FC<BivariateTabProps> = ({
    profile,
    downloadChart,
    scatterX,
    setScatterX,
    scatterY,
    setScatterY,
    scatterZ,
    setScatterZ,
    scatterColor,
    setScatterColor,
    is3D,
    setIs3D
}) => {
    return (
        <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                    <ScatterIcon className="w-5 h-5 mr-2 text-blue-500" />
                    Bivariate Analysis (Scatter Plot)
                    <InfoTooltip text="Visualize the relationship between two numeric variables." />
                </h3>
                <button
                    onClick={() => downloadChart('bivariate-chart', 'bivariate-analysis', 'Bivariate Analysis', `${scatterX} vs ${scatterY}`)}
                    className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                    title="Download Chart"
                >
                    <Download className="w-4 h-4" />
                </button>
            </div>
            
            <div className="flex gap-4 mb-6 items-end">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">X Axis</label>
                    <select 
                        value={scatterX} 
                        onChange={(e) => setScatterX(e.target.value)}
                        className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
                    >
                        <option value="">Select Variable</option>
                        {Object.values(profile.columns)
                            .filter((c: any) => c.dtype === 'Numeric')
                            .map((c: any) => (
                                <option key={c.name} value={c.name}>{c.name}</option>
                            ))
                        }
                    </select>
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Y Axis</label>
                    <select 
                        value={scatterY} 
                        onChange={(e) => setScatterY(e.target.value)}
                        className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
                    >
                        <option value="">Select Variable</option>
                        {Object.values(profile.columns)
                            .filter((c: any) => c.dtype === 'Numeric')
                            .map((c: any) => (
                                <option key={c.name} value={c.name}>{c.name}</option>
                            ))
                        }
                    </select>
                </div>
                
                {is3D && (
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Z Axis</label>
                        <select 
                            value={scatterZ} 
                            onChange={(e) => setScatterZ(e.target.value)}
                            className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
                        >
                            <option value="">Select Variable</option>
                            {Object.values(profile.columns)
                                .filter((c: any) => c.dtype === 'Numeric')
                                .map((c: any) => (
                                    <option key={c.name} value={c.name}>{c.name}</option>
                                ))
                            }
                        </select>
                    </div>
                )}

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Color By</label>
                    <select 
                        value={scatterColor} 
                        onChange={(e) => setScatterColor(e.target.value)}
                        className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
                    >
                        <option value="">None</option>
                        {Object.values(profile.columns)
                            .map((c: any) => (
                                <option key={c.name} value={c.name}>{c.name}</option>
                            ))
                        }
                    </select>
                </div>

                <button
                    onClick={() => setIs3D(!is3D)}
                    className={`p-2 rounded-md border transition-colors ${
                        is3D 
                            ? 'bg-blue-50 border-blue-200 text-blue-700 dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-300' 
                            : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300'
                    }`}
                    title="Toggle 3D View"
                >
                    <Box className="w-5 h-5" />
                </button>
            </div>

            <div id="bivariate-chart">
            {scatterX && scatterY ? (
                is3D && scatterZ ? (
                    <ThreeDScatterPlot 
                        data={profile.sample_data} 
                        xKey={scatterX} 
                        yKey={scatterY} 
                        zKey={scatterZ}
                        labelKey={scatterColor || undefined}
                        xLabel={scatterX}
                        yLabel={scatterY}
                        zLabel={scatterZ}
                    />
                ) : (
                    <CanvasScatterPlot 
                        data={profile.sample_data} 
                        xKey={scatterX} 
                        yKey={scatterY} 
                        labelKey={scatterColor || undefined}
                        xLabel={scatterX}
                        yLabel={scatterY}
                    />
                )
            ) : (
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-900 rounded text-gray-400 text-sm">
                    Select X and Y variables to generate scatter plot.
                </div>
            )}
            </div>
        </div>
    );
};