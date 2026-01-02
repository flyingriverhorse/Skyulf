import React from 'react';
import Plot from 'react-plotly.js';

interface ThreeDScatterPlotProps {
  data: any[];
  xKey: string;
  yKey: string;
  zKey: string;
  labelKey?: string;
  xLabel?: string;
  yLabel?: string;
  zLabel?: string;
  height?: number;
}

export const ThreeDScatterPlot: React.FC<ThreeDScatterPlotProps> = ({
  data,
  xKey,
  yKey,
  zKey,
  labelKey,
  xLabel,
  yLabel,
  zLabel,
  height = 600
}) => {
  
  // Prepare traces
  const traces: any[] = [];
  
  if (labelKey) {
    // Group by label
    const groups: {[key: string]: any[]} = {};
    data.forEach(d => {
      const label = d[labelKey] || 'Other';
      if (!groups[label]) groups[label] = [];
      groups[label].push(d);
    });
    
    Object.keys(groups).forEach((label) => {
      const groupData = groups[label];
      traces.push({
        x: groupData.map(d => d[xKey]),
        y: groupData.map(d => d[yKey]),
        z: groupData.map(d => d[zKey]),
        mode: 'markers',
        type: 'scatter3d',
        name: label,
        marker: {
          size: 3,
          opacity: 0.8
        },
        hovertemplate: 
            `<b>${label}</b><br>` +
            `${xLabel || xKey}: %{x}<br>` +
            `${yLabel || yKey}: %{y}<br>` +
            `${zLabel || zKey}: %{z}<extra></extra>`
      });
    });
  } else {
    traces.push({
      x: data.map(d => d[xKey]),
      y: data.map(d => d[yKey]),
      z: data.map(d => d[zKey]),
      mode: 'markers',
      type: 'scatter3d',
      name: 'Data Points',
      marker: {
        size: 3,
        color: '#8884d8',
        opacity: 0.8
      },
      hovertemplate: 
        `${xLabel || xKey}: %{x}<br>` +
        `${yLabel || yKey}: %{y}<br>` +
        `${zLabel || zKey}: %{z}<extra></extra>`
    });
  }

  return (
    <div style={{ height: height, width: '100%' }}>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: height,
          margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 0
          },
          scene: {
            xaxis: { title: { text: xLabel || xKey } },
            yaxis: { title: { text: yLabel || yKey } },
            zaxis: { title: { text: zLabel || zKey } },
          },
          showlegend: true,
          legend: {
            x: 0,
            y: 1
          }
        } as any}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
        config={{ displayModeBar: true }}
      />
    </div>
  );
};
