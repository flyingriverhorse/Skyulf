import { getSmoothStepPath, getStraightPath, Position, type EdgeProps, type useReactFlow } from '@xyflow/react';

export interface EdgePoint { x: number; y: number }
interface SplitPoint extends EdgePoint { id: string }
type GetInternalNode = ReturnType<typeof useReactFlow>['getInternalNode'];

/** Resolve handle centers in canvas coordinates, ignoring unavailable handles. */
function getSplitPoints(props: EdgeProps, getInternalNode: GetInternalNode): SplitPoint[] {
  const handles: string[] = Array.isArray(props.data?.splitHandles) ? props.data.splitHandles : [];
  const node = handles.length ? getInternalNode(props.source) : undefined;
  return handles.flatMap(handleId => {
    const handle = node?.internals.handleBounds?.source?.find(port => port.id === handleId);
    return handle && node ? [{ id: handleId,
      x: node.internals.positionAbsolute.x + handle.x + handle.width / 2,
      y: node.internals.positionAbsolute.y + handle.y + handle.height / 2 }] : [];
  });
}

/** Place the shared trunk beyond the source and halfway between split handles. */
function getTrunkSource(props: EdgeProps, points: SplitPoint[]): EdgePoint {
  if (points.length <= 1) return { x: props.sourceX, y: props.sourceY };
  const ys = points.map(point => point.y);
  return {
    x: props.sourceX + Math.min(64, Math.max(24, (props.targetX - props.sourceX) * 0.45)),
    y: (Math.min(...ys) + Math.max(...ys)) / 2,
  };
}

/** Smooth steps can degenerate near collinear handles; mixed axes stay curved. */
function shouldUseStraight(props: EdgeProps, source: EdgePoint): boolean {
  const horizontal = [Position.Left, Position.Right];
  const vertical = [Position.Top, Position.Bottom];
  if (horizontal.includes(props.sourcePosition) && horizontal.includes(props.targetPosition)) {
    return Math.abs(props.targetY - source.y) < 6;
  }
  if (vertical.includes(props.sourcePosition) && vertical.includes(props.targetPosition)) {
    return Math.abs(props.targetX - source.x) < 6;
  }
  return false;
}

/** Compute the rendered trunk and label position from the same geometry. */
export function getEdgeGeometry(props: EdgeProps, getInternalNode: GetInternalNode) {
  const splitPoints = getSplitPoints(props, getInternalNode);
  const trunkSource = getTrunkSource(props, splitPoints);
  const coordinates = { sourceX: trunkSource.x, sourceY: trunkSource.y, targetX: props.targetX, targetY: props.targetY };
  const [edgePath, labelX, labelY] = shouldUseStraight(props, trunkSource)
    ? getStraightPath(coordinates)
    : getSmoothStepPath({ ...coordinates, sourcePosition: props.sourcePosition, targetPosition: props.targetPosition, borderRadius: 16 });
  return { splitPoints, grouped: splitPoints.length > 1, trunkSource, edgePath, labelX, labelY };
}

export type EdgeGeometry = ReturnType<typeof getEdgeGeometry>;
