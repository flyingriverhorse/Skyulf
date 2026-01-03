import React from 'react';
import { 
    Calendar, 
    Download, 
    Info 
} from 'lucide-react';
import { 
    ResponsiveContainer, 
    LineChart, 
    CartesianGrid, 
    XAxis, 
    YAxis, 
    Tooltip, 
    Legend, 
    Line, 
    BarChart, 
    Bar, 
    ReferenceLine 
} from 'recharts';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { COLORS } from '../constants';

interface TimeSeriesTabProps {
    profile: any;
    downloadChart: (id: string, filename: string, title: string, subtitle?: string, extraInfo?: string) => void;
}

export const TimeSeriesTab: React.FC<TimeSeriesTabProps> = ({
    profile,
    downloadChart
}) => {
    return (
        <div className="mt-4 space-y-6">
            {/* Trend Chart */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center justify-between">
                    <div className="flex items-center">
                        <Calendar className="w-5 h-5 mr-2 text-blue-500" />
                        Trend Analysis ({profile.timeseries.date_col})
                        <InfoTooltip text="Shows how values change over time. Look for long-term trends (up/down) or sudden shifts." />
                    </div>
                    <div className="flex items-center gap-2">
                        {profile.timeseries.stationarity_test && (
                            <div className={`text-xs px-3 py-1 rounded-full border flex items-center gap-1 ${
                                profile.timeseries.stationarity_test.is_stationary 
                                    ? 'bg-green-50 border-green-200 text-green-700 dark:bg-green-900/20 dark:border-green-800 dark:text-green-300' 
                                    : 'bg-yellow-50 border-yellow-200 text-yellow-700 dark:bg-yellow-900/20 dark:border-yellow-800 dark:text-yellow-300'
                            }`}>
                                <InfoTooltip text="Augmented Dickey-Fuller Test. Stationary (p<0.05) means the data has constant mean/variance over time. Non-Stationary (p>=0.05) means it has a trend or seasonality." />
                                <span className="font-semibold">ADF Test:</span> {profile.timeseries.stationarity_test.is_stationary ? 'Stationary' : 'Non-Stationary'} 
                                <span className="opacity-75 ml-1">(p={profile.timeseries.stationarity_test.p_value.toFixed(3)})</span>
                            </div>
                        )}
                        <button
                            onClick={() => downloadChart(
                                'trend-chart', 
                                'trend-analysis', 
                                'Trend Analysis', 
                                `Date Column: ${profile.timeseries.date_col}`,
                                profile.timeseries.stationarity_test ? 
                                    `ADF Test: ${profile.timeseries.stationarity_test.is_stationary ? 'Stationary' : 'Non-Stationary'} (p=${profile.timeseries.stationarity_test.p_value.toFixed(3)})` : undefined
                            )}
                            className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                            title="Download Chart"
                        >
                            <Download className="w-4 h-4" />
                        </button>
                    </div>
                </h3>
                <div className="h-80 w-full" id="trend-chart">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={profile.timeseries.trend}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis 
                                dataKey="date" 
                                tick={{ fontSize: 12 }} 
                                minTickGap={30}
                            />
                            <YAxis />
                            <Tooltip 
                                contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                labelStyle={{ color: '#9ca3af' }}
                            />
                            <Legend />
                            {Object.keys(profile.timeseries.trend[0]?.values || {}).map((key, idx) => (
                                <Line 
                                    key={key}
                                    type="monotone" 
                                    dataKey={`values.${key}`} 
                                    name={key}
                                    stroke={COLORS[idx % COLORS.length]} 
                                    dot={false}
                                    strokeWidth={2}
                                />
                            ))}
                            {/* Fallback if no values (just count) */}
                            {(!profile.timeseries.trend[0]?.values || Object.keys(profile.timeseries.trend[0].values).length === 0) && (
                                    <Line type="monotone" dataKey="count" stroke="#3b82f6" dot={false} />
                            )}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Seasonality Grid */}
            <div className="grid grid-cols-1 gap-6">
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm relative group">
                    <div className="absolute top-4 right-4 opacity-100 z-10">
                        <button
                            onClick={() => downloadChart('day-seasonality-chart', 'day-seasonality', 'Day of Week Seasonality')}
                            className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                            title="Download Chart"
                        >
                            <Download className="w-3 h-3" />
                        </button>
                    </div>
                    <div className="flex items-center mb-4">
                        <h3 className="text-sm font-medium text-gray-500">Day of Week Seasonality</h3>
                        <InfoTooltip text="Average values for each day. Helps identify weekly patterns (e.g., lower on weekends)." />
                    </div>
                    <div className="h-64 w-full" id="day-seasonality-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={profile.timeseries.seasonality.day_of_week}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                                <YAxis />
                                <Tooltip 
                                    contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                    cursor={{fill: 'transparent'}}
                                />
                                <Bar dataKey="count" fill="#8884d8" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Monthly Seasonality (Full Width) */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm relative group">
                <div className="absolute top-4 right-4 opacity-100 z-10">
                    <button
                        onClick={() => downloadChart('month-seasonality-chart', 'month-seasonality', 'Monthly Seasonality')}
                        className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                        title="Download Chart"
                    >
                        <Download className="w-3 h-3" />
                    </button>
                </div>
                <div className="flex items-center mb-4">
                    <h3 className="text-sm font-medium text-gray-500">Monthly Seasonality</h3>
                    <InfoTooltip text="Average values for each month. Helps identify yearly patterns or seasonal effects." />
                </div>
                <div className="h-64 w-full" id="month-seasonality-chart">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={profile.timeseries.seasonality.month_of_year}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                            <YAxis />
                            <Tooltip 
                                contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                cursor={{fill: 'transparent'}}
                            />
                            <Bar dataKey="count" fill="#82ca9d" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Autocorrelation (ACF) */}
            {profile.timeseries.autocorrelation && profile.timeseries.autocorrelation.length > 0 && (
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm relative group">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center">
                            <h3 className="text-sm font-medium text-gray-500">Autocorrelation (Lag Analysis)</h3>
                            <div className="group relative ml-2">
                                <Info className="w-4 h-4 text-gray-400 cursor-help" />
                                <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none text-center">
                                    Shows how much the current value depends on past values (lags). 
                                    High bars indicate repeating patterns or seasonality.
                                    <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-800"></div>
                                </div>
                            </div>
                        </div>
                        <button
                            onClick={() => downloadChart('autocorrelation-chart', 'autocorrelation', 'Autocorrelation (Lag Analysis)')}
                            className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                            title="Download Chart"
                        >
                            <Download className="w-3 h-3" />
                        </button>
                    </div>
                    <div className="h-80 w-full pb-6" id="autocorrelation-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={profile.timeseries.autocorrelation} margin={{ bottom: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                    <XAxis 
                                        dataKey="lag" 
                                        tick={{ fontSize: 12 }} 
                                        label={{ value: 'Lag (Days)', position: 'insideBottom', offset: -15, fontSize: 12 }}
                                    />
                                    <YAxis domain={[-1, 1]} />
                                    <Tooltip 
                                        contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                        cursor={{fill: 'transparent'}}
                                        formatter={(value: number) => [value.toFixed(3), 'Correlation']}
                                    />
                                    <ReferenceLine y={0} stroke="#9ca3af" />
                                    <Bar dataKey="corr" fill="#f59e0b" radius={[2, 2, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}
        </div>
    );
};