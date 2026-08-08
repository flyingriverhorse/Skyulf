import React, { useMemo } from 'react';
import { Plot } from '../../core/plotly';
import { CHART_SERIES_COLORS } from '../../core/theme/chartTheme';
import { useChartTheme } from '../../core/hooks/useChartTheme';
import { groupScatterPoints } from './scatterGrouping';
import { markerShapeForIndex, toPlotlyMarkerSymbol } from './chartMarkerShapes';

/** A single point for the 3-D scatter; values are looked up by `xKey/yKey/zKey/labelKey`. */
export type ScatterPoint = Record<string, string | number | null | undefined>;

interface ThreeDScatterPlotProps {
  data: ScatterPoint[];
  xKey: string;
  yKey: string;
  zKey: string;
  labelKey?: string | undefined;
  xLabel?: string | undefined;
  yLabel?: string | undefined;
  zLabel?: string | undefined;
  height?: number | undefined;
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

  // Plotly trace shapes are wide unions; we keep them loose intentionally.
  // Groups get a color AND a marker symbol (see `chartMarkerShapes`) so
  // identity survives grayscale/color-vision-deficiency contexts.
  const traces: any[] = useMemo(() => {
    const groups = groupScatterPoints(data, labelKey);

    return Object.keys(groups).map((label, idx) => {
      const groupData = groups[label] ?? [];
      const color = CHART_SERIES_COLORS[idx % CHART_SERIES_COLORS.length]!;
      const symbol = toPlotlyMarkerSymbol(markerShapeForIndex(idx));
      return {
        x: groupData.map((d) => d[xKey]),
        y: groupData.map((d) => d[yKey]),
        z: groupData.map((d) => d[zKey]),
        mode: 'markers',
        type: 'scatter3d',
        name: label,
        marker: {
          size: 3,
          opacity: 0.8,
          color,
          symbol
        },
        hovertemplate:
            `<b>${label}</b><br>` +
            `${xLabel || xKey}: %{x}<br>` +
            `${yLabel || yKey}: %{y}<br>` +
            `${zLabel || zKey}: %{z}<extra></extra>`
      };
    });
  }, [data, xKey, yKey, zKey, labelKey, xLabel, yLabel, zLabel]);

  const theme = useChartTheme();

  if (data.length === 0) {
    return <div className="text-gray-500 text-sm italic p-4 text-center">No data points to display</div>;
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
          paper_bgcolor: theme.bgColor,
          plot_bgcolor: theme.bgColor,
          font: { color: theme.textColor },
          scene: {
            xaxis: { title: { text: xLabel || xKey }, color: theme.axisColor, gridcolor: theme.gridColor },
            yaxis: { title: { text: yLabel || yKey }, color: theme.axisColor, gridcolor: theme.gridColor },
            zaxis: { title: { text: zLabel || zKey }, color: theme.axisColor, gridcolor: theme.gridColor },
          },
          // Legend rendering is delegated to the shared, always-visible
          // `ChartLegend` alongside this chart (see `scatterGrouping.ts`),
          // so 2D and 3D scatter plots present a consistent legend UX.
          showlegend: false
        } as any}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
        config={{ displayModeBar: true }}
      />
    </div>
  );
};
