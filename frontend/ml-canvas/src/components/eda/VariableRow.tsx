import React, { useRef, useState } from 'react';
import { toPng } from 'html-to-image';
import {
    ChevronDown,
    ChevronRight,
    Download,
    Loader2,
    Check
} from 'lucide-react';
import { clickableProps } from '../../core/utils/a11y';
import { DistributionChart, type DistributionDatum } from './DistributionChart';
import { Button } from '../ui/button';
import { toast } from '../../core/toast';
import type { ColumnProfile } from '../../core/types/edaProfile';
import { VariableSummary, VariableIdentity, VariableControls } from './variableRow/VariablePresentation';
import { VariableStatistics } from './variableRow/VariableStatistics';

export interface VariableRowProps {
    profile: ColumnProfile;
    isExpanded: boolean;
    onToggleExpand: () => void;
    onToggleExclude: (colName: string, exclude: boolean) => void;
    isExcluded: boolean;
    handleAddFilter: (column: string, value: string | number, operator: string) => void;
}

export const VariableRow: React.FC<VariableRowProps> = ({
    profile,
    isExpanded,
    onToggleExpand,
    onToggleExclude,
    isExcluded,
    handleAddFilter
}) => {
    const chartRef = useRef<HTMLDivElement>(null);
    const [dlState, setDlState] = useState<'idle' | 'downloading' | 'done'>('idle');

    const downloadChart = async () => {
        if (!chartRef.current) return;
        setDlState('downloading');

        try {
            const isDark = document.documentElement.classList.contains('dark');
            const dataUrl = await toPng(chartRef.current, {
                backgroundColor: isDark ? '#1f2937' : '#ffffff',
                pixelRatio: 2,
            });
            const link = document.createElement('a');
            link.href = dataUrl;
            link.download = `${profile.name}_distribution.png`;
            link.click();
        } catch (error) {
            console.error('Failed to download chart', error);
            toast.error('Chart download failed', String(error));
        } finally {
            setDlState('done');
            setTimeout(() => setDlState('idle'), 1200);
        }
    };

    const onBarClick = (data: DistributionDatum) => {
        if (profile.dtype === 'Categorical' || profile.dtype === 'Boolean') {
             const val = data.value !== undefined ? data.value : data.name; // Fallback to name if value missing
             handleAddFilter(profile.name, val, '==');
        } else if (profile.dtype === 'Numeric') {
             if (data.rawBin) {
                 // Try to add >= start
                 handleAddFilter(profile.name, data.rawBin.start, '>=');
             }
        }
    };

    return (
        <div className={`border rounded-lg mb-2 transition-all ${isExpanded ? 'border-primary/50 shadow-md bg-card/50' : 'border-border bg-card'}`}>
            <div
                className="flex items-center p-3 hover:bg-accent/50 cursor-pointer rounded-t-lg"
                {...clickableProps(onToggleExpand)}
            >
                <div className="mr-2 text-muted-foreground">
                    {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                </div>

                <div className="flex-1 min-w-0">
                    <VariableIdentity profile={profile} />

                    {!isExpanded && <VariableSummary profile={profile} />}
                </div>

                <VariableControls profile={profile} isExcluded={isExcluded} onToggleExclude={onToggleExclude} />
            </div>

            {isExpanded && (
                <div className="p-4 border-t border-border bg-card/30 animate-in slide-in-from-top-2 duration-200">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch">

                        <VariableStatistics profile={profile} />

                        {/* Distribution Chart Column */}
                        <div className="md:col-span-2 flex flex-col min-h-[300px]">
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Distribution</h4>
                                <Button size="icon" variant="outline" className="h-8 w-8" title="Download Chart" onClick={downloadChart} disabled={dlState !== 'idle'}>
                                    {dlState === 'downloading' ? <Loader2 size={16} className="animate-spin" /> : dlState === 'done' ? <Check size={16} className="text-green-500" /> : <Download size={16} />}
                                </Button>
                            </div>
                            <div className="flex-1 bg-background rounded-md border p-4 min-h-0" ref={chartRef}>
                                <DistributionChart profile={profile} onBarClick={onBarClick} />
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
