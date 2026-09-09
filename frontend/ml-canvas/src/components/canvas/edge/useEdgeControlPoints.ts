import { useLayoutEffect, useRef, useState } from 'react';
import type { EdgePoint } from './edgeGeometry';

interface ControlPoints { path: string; remove: EdgePoint; warning: EdgePoint }

/** Prefer the first point with room for two upright 24px controls on a bend. */
function findWarningPoint(path: SVGPathElement, midpoint: number, remove: EdgePoint): EdgePoint {
  const spacing = 28;
  const available = Math.min(midpoint, spacing * 4);
  let warning = remove;
  let separation = 0;
  for (let offset = Math.min(spacing, available); offset < available + 4; offset += 4) {
    const arcOffset = Math.min(offset, available);
    for (const direction of [1, -1]) {
      const point = path.getPointAtLength(midpoint + direction * arcOffset);
      const distance = Math.hypot(point.x - remove.x, point.y - remove.y);
      if (distance > separation) {
        warning = point;
        separation = distance;
      }
      if (separation >= spacing) break;
    }
    if (separation >= spacing) break;
  }
  return warning;
}

/** Measurement is optional in non-SVG environments and for degenerate paths. */
function measureControls(path: SVGPathElement | null, edgePath: string): ControlPoints | null {
  if (!path || typeof path.getTotalLength !== 'function' || typeof path.getPointAtLength !== 'function') return null;
  const length = path.getTotalLength();
  if (!Number.isFinite(length) || length <= 0) return null;
  const remove = path.getPointAtLength(length / 2);
  const warning = findWarningPoint(path, length / 2, remove);
  return { path: edgePath, remove: { x: remove.x, y: remove.y }, warning: { x: warning.x, y: warning.y } };
}

/** Never position controls using measurements from an earlier edge path. */
export function useEdgeControlPoints(edgePath: string, labelX: number, labelY: number) {
  const measurementPathRef = useRef<SVGPathElement>(null);
  const [controlPoints, setControlPoints] = useState<ControlPoints | null>(null);
  useLayoutEffect(() => {
    const measured = measureControls(measurementPathRef.current, edgePath);
    if (measured) setControlPoints(measured);
  }, [edgePath]);
  const measuredPoints = controlPoints?.path === edgePath ? controlPoints : null;
  const deletePoint = measuredPoints?.remove ?? { x: labelX, y: labelY };
  const warningPoint = measuredPoints?.warning ?? deletePoint;
  return { measurementPathRef, deletePoint, warningPoint };
}
