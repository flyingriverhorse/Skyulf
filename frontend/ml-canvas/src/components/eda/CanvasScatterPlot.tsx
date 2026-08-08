import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';
import { COLORS } from './constants';
import { useChartTheme } from '../../core/hooks/useChartTheme';
import { groupScatterPoints } from './scatterGrouping';
import { markerShapeForIndex, toChartJsPointStyle } from './chartMarkerShapes';
import type { ScatterPoint } from './ThreeDScatterPlot';

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);

interface CanvasScatterPlotProps {
  data: ScatterPoint[];
  xKey: string;
  yKey: string;
  labelKey?: string | undefined; // For coloring by group
  xLabel?: string | undefined;
  yLabel?: string | undefined;
  height?: number | undefined;
}

interface ChartPoint {
  x: number;
  y: number;
  raw: ScatterPoint;
}

interface ChartDataset {
  label: string;
  data: ChartPoint[];
  backgroundColor: string;
  pointStyle: string;
  pointRadius: number;
  pointHoverRadius: number;
}

export const CanvasScatterPlot: React.FC<CanvasScatterPlotProps> = ({
  data,
  xKey,
  yKey,
  labelKey,
  xLabel,
  yLabel,
  height = 500
}) => {

  // Prepare datasets. Groups are colored AND shaped (see `chartMarkerShapes`)
  // so group identity survives grayscale/color-vision-deficiency contexts.
  const datasets: ChartDataset[] = useMemo(() => {
    const groups = groupScatterPoints(data, labelKey);

    return Object.keys(groups).map((label, idx) => ({
      label,
      data: (groups[label] ?? []).map((d) => ({ x: Number(d[xKey]), y: Number(d[yKey]), raw: d as ScatterPoint })),
      backgroundColor: COLORS[idx % COLORS.length]!,
      pointStyle: toChartJsPointStyle(markerShapeForIndex(idx)),
      pointRadius: 3,
      pointHoverRadius: 5,
    }));
  }, [data, xKey, yKey, labelKey]);

  const theme = useChartTheme();

  const options: ChartOptions<'scatter'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: {
          display: !!xLabel,
          text: xLabel,
          color: theme.axisColor,
          font: {
            weight: 'bold'
          }
        },
        grid: {
            color: theme.gridColor,
            lineWidth: 1
        },
        ticks: {
            color: theme.axisColor
        }
      },
      y: {
        title: {
          display: !!yLabel,
          text: yLabel,
          color: theme.axisColor,
          font: {
            weight: 'bold'
          }
        },
        grid: {
            color: theme.gridColor,
            lineWidth: 1
        },
        ticks: {
            color: theme.axisColor
        }
      }
    },
    plugins: {
      tooltip: {
        callbacks: {
          // chart.js TooltipModel callback signature is wide; we keep this loose intentionally.
          label: (context: any) => {
            const point = context.raw;
            let label = context.dataset.label || '';
            if (label) label += ': ';
            label += `(${point.x}, ${point.y})`;
            return label;
          }
        }
      },
      legend: {
        // The scrollable/filterable `ChartLegend` rendered alongside this
        // chart is the persistent legend; Chart.js's own legend is disabled
        // here so groups are never silently hidden past a group-count cutoff.
        display: false
      }
    }
  };

  return (
    <div style={{ height: height, width: '100%' }}>
      <Scatter options={options} data={{ datasets }} />
    </div>
  );
};
