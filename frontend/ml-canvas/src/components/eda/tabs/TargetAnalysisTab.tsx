import React from 'react';
import { 
    BarChart2, 
    Download, 
    ScatterChart as ScatterIcon, 
    Clock, 
    Loader2 
} from 'lucide-react';
import { 
    ResponsiveContainer, 
    BarChart, 
    CartesianGrid, 
    XAxis, 
    YAxis, 
    Tooltip, 
    ReferenceLine, 
    Bar, 
    ComposedChart, 
    Scatter 
} from 'recharts';
import { InfoTooltip } from '../../ui/InfoTooltip';

interface TargetAnalysisTabProps {
    profile: any;
    downloadChart: (id: string, filename: string, title: string, subtitle?: string) => void;
    history: any[];
    loading: boolean;
    loadSpecificReport: (id: number) => void;
    report: any;
}

export const TargetAnalysisTab: React.FC<TargetAnalysisTabProps> = ({
    profile,
    downloadChart,
    history,
    loading,
    loadSpecificReport,
    report
}) => {
    const downloadAllInteractions = async () => {
        if (!profile?.target_interactions) return;
        
        const interactions = profile.target_interactions;
        const chartHeight = 400; // Height per chart
        const chartWidth = 800;
        const padding = 40;
        
        // 2 Columns Layout
        const cols = 2;
        const rows = Math.ceil(interactions.length / cols);
        
        const totalWidth = (cols * chartWidth) + ((cols + 1) * padding);
        const totalHeight = (rows * (chartHeight + padding)) + 100; // +100 for main title
        
        const canvas = document.createElement('canvas');
        canvas.width = totalWidth;
        canvas.height = totalHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Background
        const isDarkMode = document.documentElement.classList.contains('dark');
        const bgColor = isDarkMode ? '#1f2937' : '#ffffff';
        const textColor = isDarkMode ? '#ffffff' : '#111827';
        const subTextColor = isDarkMode ? '#9ca3af' : '#4b5563';

        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Main Title
        ctx.fillStyle = textColor;
        ctx.font = 'bold 24px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Detailed Interactions (Box Plots)', totalWidth / 2, 50);

        // Draw each chart
        for (let i = 0; i < interactions.length; i++) {
            const interaction = interactions[i];
            const elementId = `interaction-chart-${i}`;
            const element = document.getElementById(elementId);
            if (!element) continue;

            const svg = element.querySelector('svg');
            if (!svg) continue;
            
            const row = Math.floor(i / cols);
            const col = i % cols;
            
            const x = padding + (col * (chartWidth + padding));
            const y = 100 + (row * (chartHeight + padding));

            // Title for this chart
            ctx.fillStyle = subTextColor;
            ctx.font = 'bold 16px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(`${interaction.feature} vs ${profile.target_col}`, x + 40, y);
            
            if (interaction.p_value !== undefined) {
                ctx.font = '14px sans-serif';
                ctx.fillStyle = interaction.p_value < 0.05 ? '#059669' : subTextColor;
                ctx.fillText(`ANOVA p: ${Number(interaction.p_value).toExponential(2)}`, x + 40, y + 20);
            }

            // Serialize SVG
            const serializer = new XMLSerializer();
            const svgString = serializer.serializeToString(svg);
            const img = new Image();
            const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
            const url = URL.createObjectURL(svgBlob);

            await new Promise<void>((resolve) => {
                img.onload = () => {
                    ctx.drawImage(img, x, y + 30, chartWidth, chartHeight - 50); // Adjust height to fit
                    URL.revokeObjectURL(url);
                    resolve();
                };
                img.src = url;
            });
        }

        // Download
        const link = document.createElement('a');
        link.download = 'all-interactions.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    };

    return (
        <div className="mt-4 grid grid-cols-1 lg:grid-cols-4 gap-6">
            <div className="lg:col-span-3 space-y-6">
                {/* Main Analysis Content */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                            <BarChart2 className="w-5 h-5 mr-2 text-blue-500" />
                            Target Analysis: <span className="ml-2 text-blue-600">{profile.target_col}</span>
                            <InfoTooltip text="Analyzes relationships between the target variable and other features. Shows top correlations and key drivers." />
                        </h2>
                        <button
                            onClick={() => downloadChart('target-analysis-chart', 'target-analysis', `Target Analysis: ${profile.target_col}`, 'Top Associated Features')}
                            className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                            title="Download Chart"
                        >
                            <Download className="w-4 h-4" />
                        </button>
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <div className="lg:col-span-2">
                            <h3 className="text-sm font-medium text-gray-500 mb-4">Top Associated Features</h3>
                            <div className="h-64 w-full" id="target-analysis-chart">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart 
                                        data={Object.entries(profile.target_correlations)
                                            .slice(0, 10)
                                            .map(([k, v]) => ({ name: k, value: v as number }))
                                        } 
                                        layout="vertical"
                                        margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                        <XAxis type="number" domain={[0, 1]} />
                                        <YAxis type="category" dataKey="name" width={120} tick={{fontSize: 12}} />
                                        <Tooltip 
                                            contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                            itemStyle={{ color: '#fff' }}
                                            formatter={(value: number) => [value.toFixed(3), 'Association']}
                                        />
                                        <ReferenceLine x={0} stroke="#9ca3af" />
                                        <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={20} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                            <h3 className="text-sm font-medium text-gray-500 mb-2">Key Insights</h3>
                            <ul className="space-y-2 text-sm">
                                <li className="flex items-start">
                                    <span className="mr-2 text-blue-500">•</span>
                                    <span>Target is <strong>{profile.columns[profile.target_col]?.dtype}</strong></span>
                                </li>
                                <li className="flex items-start">
                                    <span className="mr-2 text-blue-500">•</span>
                                    <span>
                                        Strongest predictor: <strong>{Object.keys(profile.target_correlations)[0] || 'None'}</strong> 
                                        {Object.values(profile.target_correlations)[0] !== undefined && (
                                            <> ({(Object.values(profile.target_correlations)[0] as number).toFixed(2)})</>
                                        )}
                                    </span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Detailed Interactions (Box Plots) */}
                {profile.target_interactions && profile.target_interactions.length > 0 && (
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                        <div className="flex justify-between items-center mb-2">
                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                                <ScatterIcon className="w-5 h-5 mr-2 text-purple-500" />
                                Detailed Interactions (Box Plots)
                                <InfoTooltip text="Shows distribution of values across categories. The box represents the middle 50% of data (Q1 to Q3). The line inside is the median." />
                            </h3>
                            <button
                                onClick={downloadAllInteractions}
                                className="p-2 rounded-md border bg-white border-gray-300 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300"
                                title="Download All Charts"
                            >
                                <Download className="w-4 h-4" />
                            </button>
                        </div>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                            <span className="font-semibold">ANOVA Analysis:</span> A p-value &lt; 0.05 (highlighted in green) indicates that the feature varies significantly across the target categories, making it a strong predictor.
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {profile.target_interactions.map((interaction: any, idx: number) => (
                                <div key={idx} className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-100 dark:border-gray-800 relative group">
                                    <div className="absolute top-2 right-2 opacity-100">
                                        <button
                                            onClick={() => downloadChart(
                                                `interaction-chart-${idx}`, 
                                                `interaction-${interaction.feature}`,
                                                `${interaction.feature} vs ${profile.target_col}`,
                                                interaction.p_value !== undefined ? `ANOVA p: ${Number(interaction.p_value).toExponential(2)}` : undefined
                                            )}
                                            className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                                            title="Download Chart"
                                        >
                                            <Download className="w-3 h-3" />
                                        </button>
                                    </div>
                                    <div className="text-center mb-2">
                                        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                            {interaction.feature} vs {profile.target_col}
                                        </h4>
                                        {interaction.p_value !== undefined && interaction.p_value !== null && (
                                            <span className={`text-xs px-2 py-0.5 rounded-full inline-block mt-1 ${
                                                interaction.p_value < 0.05 
                                                    ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' 
                                                    : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                                            }`}>
                                                ANOVA p: {Number(interaction.p_value).toExponential(2)}
                                            </span>
                                        )}
                                    </div>
                                    <div className="h-64 w-full" id={`interaction-chart-${idx}`}>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <ComposedChart 
                                                data={interaction.data.map((d: any) => ({
                                                    ...d,
                                                    boxBottom: d.stats.q1,
                                                    boxHeight: d.stats.q3 - d.stats.q1
                                                }))} 
                                                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                            >
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                                                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
                                                <Tooltip 
                                                    cursor={{ fill: 'transparent' }}
                                                    content={({ active, payload }) => {
                                                        if (active && payload && payload.length) {
                                                            const data = payload[0].payload;
                                                            return (
                                                                <div className="bg-gray-800 text-white text-xs p-2 rounded shadow-lg z-50">
                                                                    <p className="font-semibold mb-1">{data.name}</p>
                                                                    <p>Max: {data.stats.max.toFixed(2)}</p>
                                                                    <p>Q3: {data.stats.q3.toFixed(2)}</p>
                                                                    <p>Median: {data.stats.median.toFixed(2)}</p>
                                                                    <p>Q1: {data.stats.q1.toFixed(2)}</p>
                                                                    <p>Min: {data.stats.min.toFixed(2)}</p>
                                                                </div>
                                                            );
                                                        }
                                                        return null;
                                                    }}
                                                />
                                                {/* Invisible bar to push the box up to Q1 */}
                                                <Bar dataKey="boxBottom" stackId="a" fill="transparent" legendType="none" />
                                                {/* The Box (Q1 to Q3) */}
                                                <Bar dataKey="boxHeight" stackId="a" fill="#8884d8" fillOpacity={0.6} name="IQR (Q1-Q3)" />
                                                
                                                {/* Median, Min, Max Points */}
                                                <Scatter name="Median" dataKey="stats.median" fill="#ff7300" shape="square" />
                                                <Scatter name="Max" dataKey="stats.max" fill="#82ca9d" shape="cross" />
                                                <Scatter name="Min" dataKey="stats.min" fill="#82ca9d" shape="cross" />
                                            </ComposedChart>
                                        </ResponsiveContainer>
                                    </div>
                                    <p className="text-xs text-center text-gray-400 mt-2">
                                        * Box: Q1-Q3. Orange square: Median. Green crosses: Min/Max.
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* History Sidebar */}
            <div className="lg:col-span-1">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm h-fit sticky top-4">
                    <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3 flex items-center justify-between">
                        <div className="flex items-center">
                            <Clock className="w-4 h-4 mr-2" />
                            Analysis History
                        </div>
                        {loading && <Loader2 className="w-3 h-3 animate-spin text-blue-500" />}
                    </h3>
                    <div className="space-y-2 max-h-[500px] overflow-y-auto">
                        {history.length === 0 ? (
                            <p className="text-xs text-gray-500 italic">No previous analyses found.</p>
                        ) : (
                            history.slice(0, 10).map((item) => (
                                <button
                                    key={item.id}
                                    onClick={() => !loading && loadSpecificReport(item.id)}
                                    disabled={loading}
                                    className={`w-full text-left p-3 rounded-md text-sm border transition-colors ${
                                        report && report.id === item.id
                                            ? 'bg-blue-50 border-blue-200 text-blue-700 dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-300'
                                            : 'bg-gray-50 border-gray-100 text-gray-600 hover:bg-gray-100 dark:bg-gray-900 dark:border-gray-800 dark:text-gray-400 dark:hover:bg-gray-800'
                                    } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                                >
                                    <div className="font-medium truncate">
                                        {item.target_col ? `Target: ${item.target_col}` : 'General Analysis'}
                                    </div>
                                    <div className="text-xs opacity-70 mt-1">
                                        {new Date(item.created_at).toLocaleString()}
                                    </div>
                                    <div className={`text-xs mt-1 inline-block px-1.5 py-0.5 rounded ${
                                        item.status === 'COMPLETED' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                                        item.status === 'FAILED' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' :
                                        'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
                                    }`}>
                                        {item.status}
                                    </div>
                                </button>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};