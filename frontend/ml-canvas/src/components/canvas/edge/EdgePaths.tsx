import type { CSSProperties, RefObject } from 'react';
import { BaseEdge, getBezierPath, Position } from '@xyflow/react';
import type { EdgeGeometry } from './edgeGeometry';
import type { EdgeInteraction } from './useEdgeInteraction';

/** Render every resolved split branch into the shared trunk's junction. */
function SplitBranches({ geometry, style, branchColor }: {
  geometry: EdgeGeometry; style: CSSProperties; branchColor: string | undefined;
}) {
  const { splitPoints, trunkSource, grouped } = geometry;
  if (!grouped) return null;
  return <>
    {splitPoints.map(point => <g key={point.id} data-split-handle={point.id}>
      <BaseEdge path={getBezierPath({ sourceX: point.x, sourceY: point.y, sourcePosition: Position.Right,
        targetX: trunkSource.x, targetY: trunkSource.y, targetPosition: Position.Left })[0]}
        style={style} interactionWidth={24} />
    </g>)}
    <circle data-split-junction="true" cx={trunkSource.x} cy={trunkSource.y} r={3} fill={branchColor || String(style.stroke || 'hsl(var(--primary))')} />
  </>;
}

/** Keep the measurement path identical to the visible trunk. */
export function EdgePaths({ geometry, interaction, edgeStyle, branchColor, markerEnd, measurementPathRef }: {
  geometry: EdgeGeometry; interaction: EdgeInteraction; edgeStyle: CSSProperties;
  branchColor: string | undefined; markerEnd: string | undefined; measurementPathRef: RefObject<SVGPathElement>;
}) {
  return <g onMouseEnter={interaction.enter} onMouseLeave={interaction.leave}>
    <SplitBranches geometry={geometry} style={edgeStyle} branchColor={branchColor} />
    <BaseEdge
      path={geometry.edgePath}
      {...(markerEnd ? { markerEnd } : {})}
      style={{
        ...edgeStyle,
        filter: interaction.showControls
          ? `drop-shadow(0 0 4px ${branchColor || 'hsl(var(--primary))'})`
          : edgeStyle.filter,
      }}
      className="react-flow__edge-path"
      interactionWidth={24}
    />
    <path ref={measurementPathRef} d={geometry.edgePath} fill="none" stroke="none" pointerEvents="none" aria-hidden="true" />
  </g>;
}
