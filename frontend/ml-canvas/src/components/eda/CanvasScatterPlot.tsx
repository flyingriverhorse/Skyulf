import React from 'react';
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
import { COLORS, getChartTheme } from './constants';

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);

interface CanvasScatterPlotProps {
  data: any[];
  xKey: string;
  yKey: string;
  labelKey?: string; // For coloring by group
  xLabel?: string;
  yLabel?: string;
  height?: number;
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
  
  // Prepare datasets
  const datasets: any[] = [];
  
  if (labelKey) {
    // Group by label
    const groups: {[key: string]: any[]} = {};
    data.forEach(d => {
      const label = d[labelKey] || 'Other';
      if (!groups[label]) groups[label] = [];
      groups[label].push({ x: d[xKey], y: d[yKey], raw: d });
    });
    
    Object.keys(groups).forEach((label, idx) => {
      datasets.push({
        label: label,
        data: groups[label],
        backgroundColor: COLORS[idx % COLORS.length],
        pointRadius: 3,
        pointHoverRadius: 5
      });
    });
  } else {
    // Single dataset
    datasets.push({
      label: 'Data Points',
      data: data.map(d => ({ x: d[xKey], y: d[yKey], raw: d })),
      backgroundColor: '#8884d8',
      pointRadius: 3,
      pointHoverRadius: 5
    });
  }

  const theme = getChartTheme();

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
        display: !!labelKey && datasets.length < 20 // Hide legend if too many groups
      }
    }
  };

  return (
    <div style={{ height: height, width: '100%' }}>
      <Scatter options={options} data={{ datasets }} />
    </div>
  );
};
