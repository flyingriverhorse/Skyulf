import React, { useState } from 'react';
import { BarChart2, Download, Loader2, Check } from 'lucide-react';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { CorrelationHeatmap } from '../CorrelationHeatmap';
import { EmptyState } from '../../shared/EmptyState';
import { getChartTheme } from '../constants';

interface CorrelationsTabProps {
    profile: any;
}

export const CorrelationsTab: React.FC<CorrelationsTabProps> = ({
    profile
}) => {
    const [activeBtn, setActiveBtn] = useState<string | null>(null);
    const [doneBtn, setDoneBtn] = useState<string | null>(null);

    if (!profile?.correlations) {
        return <EmptyState icon={<BarChart2 className="w-12 h-12 text-slate-300 dark:text-slate-600" />} title="No Correlation Data" description="Not enough numeric columns to compute correlations." />;
    }
    const downloadMatrix = async (data: any, titleText: string, filename: string) => {
        if (!data) return;
        setActiveBtn(filename);

        try {
            const theme = getChartTheme();
            const MAX_COLS = 20;
            const columns = data.columns.slice(0, MAX_COLS);
            const values = data.values.slice(0, MAX_COLS).map((row: number[]) => row.slice(0, MAX_COLS));
            
            const cellSize = 60;
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            
            ctx.font = '12px sans-serif';
            let maxLabelWidth = 0;
            columns.forEach((col: string) => {
                const w = ctx.measureText(col).width;
                if (w > maxLabelWidth) maxLabelWidth = w;
            });
            
            const labelWidth = maxLabelWidth + 40;
            const headerHeight = (maxLabelWidth * 0.7) + 60;
            const titleHeight = 60;
            
            const width = labelWidth + (columns.length * cellSize) + 50;
            const height = headerHeight + (columns.length * cellSize) + titleHeight + 50;

            canvas.width = width;
            canvas.height = height;

            ctx.fillStyle = theme.bgColor;
            ctx.fillRect(0, 0, width, height);

            ctx.fillStyle = theme.textColor;
            ctx.font = 'bold 24px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(titleText, width / 2, 40);

            const getColor = (val: number) => {
                if (val === null) return theme.bgColor === '#ffffff' ? '#f3f4f6' : '#374151';
                const opacity = Math.max(0.2, Math.abs(val));
                if (val > 0) {
                    return `rgba(239, 68, 68, ${opacity})`;
                } else {
                    return `rgba(59, 130, 246, ${opacity})`;
                }
            };

            ctx.font = '12px sans-serif';
            ctx.textBaseline = 'middle';

            values.forEach((row: number[], i: number) => {
                ctx.fillStyle = theme.textColor;
                ctx.textAlign = 'right';
                ctx.fillText(columns[i], labelWidth - 10, headerHeight + titleHeight + (i * cellSize) + (cellSize/2));

                row.forEach((val: number, j: number) => {
                    const x = labelWidth + (j * cellSize);
                    const y = headerHeight + titleHeight + (i * cellSize);

                    ctx.fillStyle = getColor(val);
                    ctx.fillRect(x, y, cellSize - 2, cellSize - 2);

                    if (val !== null) {
                        ctx.fillStyle = Math.abs(val) > 0.5 ? '#ffffff' : theme.textColor;
                        ctx.textAlign = 'center';
                        ctx.fillText(val.toFixed(2), x + (cellSize/2), y + (cellSize/2));
                    }
                });
            });

            ctx.save();
            columns.forEach((col: string, j: number) => {
                const x = labelWidth + (j * cellSize) + (cellSize/2);
                const y = headerHeight + titleHeight - 10;
                
                ctx.translate(x, y);
                ctx.rotate(-Math.PI / 4);
                ctx.fillStyle = theme.textColor;
                ctx.textAlign = 'left';
                ctx.fillText(col, 0, 0);
                ctx.rotate(Math.PI / 4);
                ctx.translate(-x, -y);
            });
            ctx.restore();

            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL('image/png');
            link.click();
        } finally {
            setActiveBtn(null);
            setDoneBtn(filename);
            setTimeout(() => setDoneBtn(null), 1200);
        }
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
                            disabled={activeBtn !== null}
                            className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm disabled:opacity-50"
                            title="Download Matrix"
                        >
                            {activeBtn === 'feature-correlations.png' ? <Loader2 className="w-4 h-4 animate-spin" /> : doneBtn === 'feature-correlations.png' ? <Check className="w-4 h-4 text-green-500" /> : <Download className="w-4 h-4" />}
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
                                disabled={activeBtn !== null}
                                className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm disabled:opacity-50"
                                title="Download Matrix"
                            >
                                {activeBtn === 'target-correlations.png' ? <Loader2 className="w-4 h-4 animate-spin" /> : doneBtn === 'target-correlations.png' ? <Check className="w-4 h-4 text-green-500" /> : <Download className="w-4 h-4" />}
                            </button>
                        </div>
                    </div>
                    <CorrelationHeatmap data={profile.correlations_with_target} />
                </div>
            )}
        </div>
    );
};