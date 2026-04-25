import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { InfoTooltip } from '../ui/InfoTooltip';
import { getChartTheme } from './constants';
import { getTooltipContentStyle } from '../../core/utils/chartUtils';
import type { ColumnProfile, HistogramBin } from '../../core/types/edaProfile';

/**
 * Shape produced by the local recharts adapter — kept narrow so the
 * `onBarClick` consumer can pull `value` for categorical bars and
 * `rawBin.start` for histogram bars without `any`.
 */
export interface DistributionDatum {
  name: string;
  fullName: string;
  count: number;
  rawBin?: HistogramBin;
  value?: string | number;
}

interface DistributionChartProps {
  profile: ColumnProfile;
  onBarClick?: (data: DistributionDatum) => void;
}

export const DistributionChart: React.FC<DistributionChartProps> = ({ profile, onBarClick }) => {
  if (!profile) return null;

  let data: DistributionDatum[] = [];
  const type = profile.dtype;

  if (type === 'Numeric' && profile.histogram) {
    data = profile.histogram.map((bin) => ({
      name: `${Number(bin.start).toFixed(2)}`, // Simplified label for X-axis
      fullName: `${Number(bin.start).toFixed(2)} - ${Number(bin.end).toFixed(2)}`,
      count: bin.count,
      rawBin: bin
    }));
  } else if (type === 'Text' && profile.histogram) {
    data = profile.histogram.map((bin) => ({
      name: `${Number(bin.start).toFixed(0)}`,
      fullName: `${Number(bin.start).toFixed(0)} - ${Number(bin.end).toFixed(0)} chars`,
      count: bin.count,
      rawBin: bin
    }));
  } else if (type === 'DateTime' && profile.histogram) {
    data = profile.histogram.map((bin) => ({
      name: new Date(bin.start).toLocaleDateString(undefined, { month: 'short', year: '2-digit' }),
      fullName: `${new Date(bin.start).toLocaleDateString()} - ${new Date(bin.end).toLocaleDateString()}`,
      count: bin.count,
      rawBin: bin
    }));
  } else if (type === 'Categorical' && profile.categorical_stats?.top_k) {
    data = profile.categorical_stats.top_k.map((item) => ({
      name: String(item.value).substring(0, 15) + (String(item.value).length > 15 ? '...' : ''),
      fullName: String(item.value),
      count: item.count,
      value: item.value
    }));
  } else {
    return <div className="text-gray-500 text-sm italic p-4 text-center">No distribution data available for this type</div>;
  }

  if (data.length === 0) {
    return <div className="text-gray-500 text-sm italic p-4 text-center">No data points</div>;
  }

  const theme = getChartTheme();

  return (
    <div className="h-full w-full min-h-[250px] relative">
      {profile.normality_test && (
        <div className="absolute top-0 right-0 bg-white/90 dark:bg-gray-800/90 p-2 rounded border border-gray-200 dark:border-gray-700 text-xs z-10 shadow-sm">
            <div className="font-semibold text-gray-700 dark:text-gray-300 mb-1 flex items-center">
                Normality Test ({profile.normality_test.test_name})
                <InfoTooltip text="Normality tests are performed on a sample of up to 5,000 rows. Shapiro-Wilk is used for smaller datasets (<5,000), and Kolmogorov-Smirnov for larger ones. For very large datasets, rely on Skewness and Kurtosis." align="right" />
            </div>
            <div className="flex justify-between gap-4">
                <span className="text-gray-500">p-value:</span>
                <span className={`font-mono ${profile.normality_test.is_normal ? 'text-green-600' : 'text-red-500'}`}>
                    {profile.normality_test.p_value.toExponential(2)}
                </span>
            </div>
            <div className="mt-1 text-[10px] text-gray-400">
                {profile.normality_test.is_normal ? 'Likely Normal Distribution' : 'Likely Not Normal'}
            </div>
        </div>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 30,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={theme.gridColor} />
          <XAxis 
            dataKey="name" 
            tick={{ fontSize: 11, fill: theme.axisColor }} 
            interval="preserveStartEnd"
            angle={-45}
            textAnchor="end"
          />
          <YAxis tick={{ fontSize: 11, fill: theme.axisColor }} />
          <Tooltip 
            content={({ active, payload }) => {
                if (active && payload && payload.length) {
                const item = payload[0].payload;
                return (
                    <div style={getTooltipContentStyle()} className="text-xs p-2 rounded shadow-lg">
                    <p className="font-semibold mb-1">{item.fullName}</p>
                    <p>Count: {item.count}</p>
                    {onBarClick && <p className="text-blue-300 mt-1 italic">Click to filter</p>}
                    </div>
                );
                }
                return null;
            }}
          />
          <Bar 
            dataKey="count" 
            radius={[4, 4, 0, 0]} 
            onClick={(d) => onBarClick && onBarClick(d as unknown as DistributionDatum)}
            className={onBarClick ? "cursor-pointer hover:opacity-80" : ""}
          >
            {data.map((_, index) => (
              <Cell key={`cell-${index}`} fill={type === 'Numeric' ? '#3b82f6' : '#8b5cf6'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
