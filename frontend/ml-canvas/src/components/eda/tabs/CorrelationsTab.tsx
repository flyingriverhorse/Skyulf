import React from 'react';
import { BarChart2, Download } from 'lucide-react';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { CorrelationHeatmap } from '../CorrelationHeatmap';

interface CorrelationsTabProps {
    profile: any;
}

export const CorrelationsTab: React.FC<CorrelationsTabProps> = ({
    profile
}) => {
    const downloadMatrix = (data: any, titleText: string, filename: string) => {
        if (!data) return;
        
        const MAX_COLS = 20;
        const columns = data.columns.slice(0, MAX_COLS);
        const values = data.values.slice(0, MAX_COLS).map((row: number[]) => row.slice(0, MAX_COLS));
        
        const cellSize = 60;
        
        // Calculate dynamic label sizes
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        ctx.font = '12px sans-serif';
        let maxLabelWidth = 0;
        columns.forEach((col: string) => {
            const w = ctx.measureText(col).width;
            if (w > maxLabelWidth) maxLabelWidth = w;
        });
        
        // Add padding
        const labelWidth = maxLabelWidth + 40; // Increased padding
        // For rotated labels, the height depends on the length of the text
        // sin(45) * width approx 0.7 * width
        const headerHeight = (maxLabelWidth * 0.7) + 60; // Increased header height
        const titleHeight = 60;
        
        const width = labelWidth + (columns.length * cellSize) + 50; // +50 padding
        const height = headerHeight + (columns.length * cellSize) + titleHeight + 50;

        canvas.width = width;
        canvas.height = height;

        // Background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);

        // Title
        ctx.fillStyle = '#111827';
        ctx.font = 'bold 24px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(titleText, width / 2, 40);

        // Helper for color
        const getColor = (val: number) => {
            if (val === null) return '#f3f4f6';
            const opacity = Math.max(0.2, Math.abs(val));
            if (val > 0) {
                return `rgba(239, 68, 68, ${opacity})`;
            } else {
                return `rgba(59, 130, 246, ${opacity})`;
            }
        };

        ctx.font = '12px sans-serif';
        ctx.textBaseline = 'middle';

        // Draw Grid
        values.forEach((row: number[], i: number) => {
            // Row Label (Right aligned)
            ctx.fillStyle = '#374151';
            ctx.textAlign = 'right';
            ctx.fillText(columns[i], labelWidth - 10, headerHeight + titleHeight + (i * cellSize) + (cellSize/2));

            row.forEach((val: number, j: number) => {
                const x = labelWidth + (j * cellSize);
                const y = headerHeight + titleHeight + (i * cellSize);

                // Cell
                ctx.fillStyle = getColor(val);
                ctx.fillRect(x, y, cellSize - 2, cellSize - 2); // -2 for gap

                // Value
                if (val !== null) {
                    ctx.fillStyle = Math.abs(val) > 0.5 ? '#ffffff' : '#000000';
                    ctx.textAlign = 'center';
                    ctx.fillText(val.toFixed(2), x + (cellSize/2), y + (cellSize/2));
                }
            });
        });

        // Column Labels (Rotated)
        ctx.save();
        columns.forEach((col: string, j: number) => {
            const x = labelWidth + (j * cellSize) + (cellSize/2);
            const y = headerHeight + titleHeight - 10;
            
            ctx.translate(x, y);
            ctx.rotate(-Math.PI / 4);
            ctx.fillStyle = '#374151';
            ctx.textAlign = 'left';
            ctx.fillText(col, 0, 0);
            ctx.rotate(Math.PI / 4);
            ctx.translate(-x, -y);
        });
        ctx.restore();

        // Download
        const link = document.createElement('a');
        link.download = filename;
        link.href = canvas.toDataURL('image/png');
        link.click();
    };

    return (
        <div className="space-y-8">
            {/* 1. Feature Correlations (Multicollinearity) */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                        <BarChart2 className="w-5 h-5 mr-2 text-blue-500" />
                        Feature Correlations (Multicollinearity)
                    </h3>
                    <div className="flex items-center gap-2">
                        <InfoTooltip text="Correlation between features only. Use this to detect redundant variables (multicollinearity)." />
                        <button
                            onClick={() => downloadMatrix(profile.correlations, "Feature Correlations", "feature-correlations.png")}
                            className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                            title="Download Matrix"
                        >
                            <Download className="w-4 h-4" />
                        </button>
                    </div>
                </div>
                <CorrelationHeatmap data={profile.correlations} />
            </div>

            {/* 2. Target Correlations (Feature Selection) */}
            {profile.correlations_with_target && (
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                            <BarChart2 className="w-5 h-5 mr-2 text-green-500" />
                            Target Correlations (Feature Selection)
                        </h3>
                        <div className="flex items-center gap-2">
                            <InfoTooltip text="Correlation including the target variable. Use this to identify the most predictive features." />
                            <button
                                onClick={() => downloadMatrix(profile.correlations_with_target, "Target Correlations", "target-correlations.png")}
                                className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                                title="Download Matrix"
                            >
                                <Download className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                    <CorrelationHeatmap data={profile.correlations_with_target} />
                </div>
            )}
        </div>
    );
};