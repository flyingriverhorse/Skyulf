import React, { useRef } from 'react';
import html2canvas from 'html2canvas';
import { 
    ChevronDown, 
    ChevronRight, 
    MoreHorizontal, 
    AlertTriangle, 
    Download,
    Eye,
    EyeOff
} from 'lucide-react';
import { BarChart, Bar, ResponsiveContainer } from 'recharts';
import { DistributionChart } from './DistributionChart';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { InfoTooltip } from '../ui/InfoTooltip';

interface VariableRowProps {
    profile: any;
    isExpanded: boolean;
    onToggleExpand: () => void;
    onToggleExclude: (colName: string, exclude: boolean) => void;
    isExcluded: boolean;
    handleAddFilter: (column: string, value: any, operator: string) => void;
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

    // Prepare mini histogram data
    let miniChartData: any[] = [];
    if (profile.histogram) {
        miniChartData = profile.histogram.map((b: any) => ({ name: String(b.start), count: b.count }));
    } else if (profile.categorical_stats?.top_k) {
        miniChartData = profile.categorical_stats.top_k.slice(0, 10).map((k: any) => ({ name: k.value, count: k.count }));
    }
    
    // Fallback for mini chart data if not directly available in standard path
    if (miniChartData.length === 0 && profile.dtype === 'Numeric' && profile.numeric_stats) {
        // Can't really plot without histogram data
    }

    const downloadChart = async () => {
        if (!chartRef.current) return;
        
        try {
            const canvas = await html2canvas(chartRef.current);
            const image = canvas.toDataURL("image/png");
            const link = document.createElement("a");
            link.href = image;
            link.download = `${profile.name}_distribution.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error("Failed to download chart", error);
        }
    };

    const onBarClick = (data: any) => {
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

    const getTypeColor = (type: string) => {
        switch (type) {
            case 'Numeric': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
            case 'Categorical': return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300';
            case 'Boolean': return 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300';
            case 'DateTime': return 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300';
            case 'Text': return 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-300';
            default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300';
        }
    };

    // Summary stats for the closed state row
    const renderSummary = () => {
        return (
            <div className="flex items-center justify-between w-full pr-4">
                <div className="flex gap-4 text-xs text-muted-foreground mr-auto">
                    {profile.numeric_stats ? (
                        <>
                            <span>Mean: <span className="font-mono text-foreground">{profile.numeric_stats.mean?.toFixed(2)}</span></span>
                            <span>Std: <span className="font-mono text-foreground">{profile.numeric_stats.std?.toFixed(2)}</span></span>
                            <span>Min: <span className="font-mono text-foreground">{profile.numeric_stats.min?.toFixed(2)}</span></span>
                            <span>Max: <span className="font-mono text-foreground">{profile.numeric_stats.max?.toFixed(2)}</span></span>
                        </>
                    ) : profile.categorical_stats ? (
                        <>
                            <span>Unique: <span className="font-mono text-foreground">{profile.categorical_stats.unique_count}</span></span>
                            {profile.categorical_stats.top_k?.[0] && (
                                <span>Top: <span className="font-mono text-foreground">{String(profile.categorical_stats.top_k[0].value).slice(0, 20)}</span></span>
                            )}
                        </>
                    ) : null}

                    {/* Normality / Status Indicators */}
                    {profile.normality_test && profile.normality_test.is_normal && (
                        <Badge variant="outline" className="text-[10px] h-4 text-purple-600 border-purple-200 bg-purple-50">
                            Normal Dist
                        </Badge>
                    )}
                    {profile.normality_test && !profile.normality_test.is_normal && (
                        <Badge variant="outline" className="text-[10px] h-4 text-amber-600 border-amber-200 bg-amber-50">
                            Not Normal
                        </Badge>
                    )}
                </div>

                {/* Mini Histogram */}
                {miniChartData.length > 0 && (
                    <div className="h-8 w-24 hidden sm:block opacity-70">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={miniChartData}>
                                <Bar 
                                    dataKey="count" 
                                    fill={profile.dtype === 'Numeric' ? '#3b82f6' : profile.dtype === 'Categorical' ? '#8b5cf6' : '#10b981'} 
                                    radius={[1, 1, 0, 0]} 
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className={`border rounded-lg mb-2 transition-all ${isExpanded ? 'border-primary/50 shadow-md bg-card/50' : 'border-border bg-card'}`}>
            <div 
                className="flex items-center p-3 hover:bg-accent/50 cursor-pointer rounded-t-lg"
                onClick={onToggleExpand}
            >
                <div className="mr-2 text-muted-foreground">
                    {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                </div>
                
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-sm truncate">{profile.name}</span>
                        <Badge variant="outline" className={`text-[10px] h-5 px-1 ${getTypeColor(profile.dtype)}`}>
                            {profile.dtype}
                        </Badge>
                        {profile.is_unique && <Badge variant="secondary" className="text-[10px] h-5">ID</Badge>}
                        {profile.is_constant && <Badge variant="destructive" className="text-[10px] h-5">Constant</Badge>}
                    </div>
                    
                    {!isExpanded && renderSummary()}
                </div>

                <div className="flex items-center gap-4 px-2">
                    {/* Missing Bar */}
                    <div className="flex flex-col items-end w-24">
                        <div className="flex items-center gap-1 mb-1">
                            <span className="text-[10px] text-muted-foreground">
                                {profile.missing_percentage > 0 ? `${profile.missing_percentage.toFixed(1)}% Missing` : '100% Present'}
                            </span>
                             <InfoTooltip text="Green bar indicates the percentage of valid (non-null) data. Full green means no missing values." align="end" size="sm" />
                        </div>
                        <div className="w-full h-1.5 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                            <div 
                                className={`h-full ${profile.missing_percentage > 0 ? 'bg-amber-400' : 'bg-green-500'}`} 
                                style={{ width: `${Math.max(100 - profile.missing_percentage, 0)}%` }}
                            />
                        </div>
                    </div>

                    <div className="h-8 w-[1px] bg-border mx-2" />

                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-muted-foreground"
                        onClick={(e) => {
                            e.stopPropagation();
                            onToggleExclude(profile.name, !isExcluded);
                        }}
                    >
                        {isExcluded ? <EyeOff size={16} /> : <Eye size={16} />}
                    </Button>
                </div>
            </div>

            {isExpanded && (
                <div className="p-4 border-t border-border bg-card/30 animate-in slide-in-from-top-2 duration-200">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        
                        {/* Stats Column */}
                        <div className="space-y-4">
                            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Statistics</h4>
                            <div className="bg-background rounded-md border p-3 text-sm space-y-2 font-mono">
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Missing</span>
                                    <span>{profile.missing_count} ({profile.missing_percentage.toFixed(2)}%)</span>
                                </div>
                                {profile.numeric_stats && (
                                    <>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Mean</span>
                                            <span>{profile.numeric_stats.mean?.toFixed(4)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Std Dev</span>
                                            <span>{profile.numeric_stats.std?.toFixed(4)}</span>
                                        </div>
                                        {/* Added Variance */}
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Variance</span>
                                            <span>{profile.numeric_stats.variance?.toFixed(4) || (Math.pow(profile.numeric_stats.std || 0, 2)).toFixed(4)}</span>
                                        </div>
                                        <div className="my-2 border-t border-dashed" />
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Min</span>
                                            <span>{profile.numeric_stats.min?.toFixed(4)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">25% (Q1)</span>
                                            <span>{profile.numeric_stats.q25?.toFixed(4)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Median</span>
                                            <span>{profile.numeric_stats.median?.toFixed(4)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">75% (Q3)</span>
                                            <span>{profile.numeric_stats.q75?.toFixed(4)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Max</span>
                                            <span>{profile.numeric_stats.max?.toFixed(4)}</span>
                                        </div>
                                        <div className="my-2 border-t border-dashed" />
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Skewness</span>
                                            <span>{profile.numeric_stats.skewness?.toFixed(4)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Kurtosis</span>
                                            <span>{profile.numeric_stats.kurtosis?.toFixed(4)}</span>
                                        </div>
                                    </>
                                )}
                                {profile.text_stats && (
                                    <>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Avg Length</span>
                                            <span>{profile.text_stats.avg_length?.toFixed(1)} chars</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Min Length</span>
                                            <span>{profile.text_stats.min_length}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Max Length</span>
                                            <span>{profile.text_stats.max_length}</span>
                                        </div>
                                        {profile.text_stats.sentiment_distribution && (
                                            <>
                                                <div className="my-2 border-t border-dashed" />
                                                <div className="text-xs font-semibold mb-1">Sentiment</div>
                                                <div className="flex h-3 rounded-full overflow-hidden w-full bg-secondary">
                                                    <div className="bg-green-500 h-full" style={{ width: `${(profile.text_stats.sentiment_distribution.positive || 0) * 100}%` }} title={`Positive: ${((profile.text_stats.sentiment_distribution.positive || 0) * 100).toFixed(1)}%`}></div>
                                                    <div className="bg-gray-400 h-full" style={{ width: `${(profile.text_stats.sentiment_distribution.neutral || 0) * 100}%` }} title={`Neutral: ${((profile.text_stats.sentiment_distribution.neutral || 0) * 100).toFixed(1)}%`}></div>
                                                    <div className="bg-red-500 h-full" style={{ width: `${(profile.text_stats.sentiment_distribution.negative || 0) * 100}%` }} title={`Negative: ${((profile.text_stats.sentiment_distribution.negative || 0) * 100).toFixed(1)}%`}></div>
                                                </div>
                                                <div className="flex justify-between text-[10px] text-muted-foreground mt-1 px-1">
                                                    <span>Pos: {((profile.text_stats.sentiment_distribution.positive || 0) * 100).toFixed(0)}%</span>
                                                    <span>Neu: {((profile.text_stats.sentiment_distribution.neutral || 0) * 100).toFixed(0)}%</span>
                                                    <span>Neg: {((profile.text_stats.sentiment_distribution.negative || 0) * 100).toFixed(0)}%</span>
                                                </div>
                                            </>
                                        )}
                                        {profile.text_stats.common_words && profile.text_stats.common_words.length > 0 && (
                                            <>
                                                 <div className="my-2 border-t border-dashed" />
                                                 <div className="text-xs font-semibold mb-1">Common Words</div>
                                                 <div className="flex flex-wrap gap-1">
                                                    {profile.text_stats.common_words.slice(0, 10).map((w: any, idx: number) => (
                                                        <Badge key={idx} variant="secondary" className="text-[10px] h-4">
                                                            {w.word || w.value} ({w.count})
                                                        </Badge>
                                                    ))}
                                                 </div>
                                            </>
                                        )}
                                    </>
                                )}
                                {profile.categorical_stats && (
                                    <>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Unique</span>
                                            <span>{profile.categorical_stats.unique_count}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Mode</span>
                                            <span className="truncate max-w-[100px]" title={String(profile.categorical_stats.top_k?.[0]?.value)}>
                                                {String(profile.categorical_stats.top_k?.[0]?.value || '-')}
                                            </span>
                                        </div>
                                        {profile.categorical_stats.top_k && profile.categorical_stats.top_k.length > 0 && (
                                            <>
                                                <div className="my-2 border-t border-dashed" />
                                                <div className="text-xs font-semibold mb-1">Top Categories</div>
                                                <div className="space-y-1">
                                                    {profile.categorical_stats.top_k.slice(0, 5).map((item: any, i: number) => (
                                                        <div key={i} className="flex justify-between text-xs">
                                                            <span className="truncate max-w-[120px]" title={String(item.value)}>{String(item.value)}</span>
                                                            <span className="text-muted-foreground">{item.count}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </>
                                        )}
                                    </>
                                )}
                            </div>
                        </div>

                        {/* Distribution Chart Column */}
                        <div className="md:col-span-2 flex flex-col h-full min-h-[300px]">
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Distribution</h4>
                                <Button size="icon" variant="outline" className="h-8 w-8" title="Download Chart" onClick={downloadChart}>
                                    <Download size={16} />
                                </Button>
                            </div>
                            <div className="flex-1 bg-background rounded-md border p-4" ref={chartRef}>
                                <DistributionChart profile={profile} onBarClick={onBarClick} />
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
