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
    
    const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#FFBB28', '#FF8042'];
    
    Object.keys(groups).forEach((label, idx) => {
      datasets.push({
        label: label,
        data: groups[label],
        backgroundColor: colors[idx % colors.length],
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

  const options: ChartOptions<'scatter'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: {
          display: !!xLabel,
          text: xLabel,
          color: '#6b7280',
          font: {
            weight: 'bold'
          }
        },
        grid: {
            color: 'rgba(0, 0, 0, 0.2)', // Darker grid lines
            lineWidth: 1
        },
        ticks: {
            color: '#6b7280'
        }
      },
      y: {
        title: {
          display: !!yLabel,
          text: yLabel,
          color: '#6b7280',
          font: {
            weight: 'bold'
          }
        },
        grid: {
            color: 'rgba(0, 0, 0, 0.2)', // Darker grid lines
            lineWidth: 1
        },
        ticks: {
            color: '#6b7280'
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
