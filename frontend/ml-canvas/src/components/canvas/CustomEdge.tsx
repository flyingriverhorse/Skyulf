import React, { memo } from 'react';
import { type EdgeProps, useReactFlow } from '@xyflow/react';
import { getEdgeGeometry } from './edge/edgeGeometry';
import { getEdgeStyle, getEndpointLabels, type EdgeBranch } from './edge/edgeAppearance';
import { useEdgeControlPoints } from './edge/useEdgeControlPoints';
import { useEdgeInteraction } from './edge/useEdgeInteraction';
import { EdgePaths } from './edge/EdgePaths';
import { EdgeLabels } from './edge/EdgeLabels';

export const CustomEdge: React.FC<EdgeProps> = memo((props) => {
  const { getInternalNode } = useReactFlow();
  const geometry = getEdgeGeometry(props, getInternalNode);
  const points = useEdgeControlPoints(geometry.edgePath, geometry.labelX, geometry.labelY);
  const interaction = useEdgeInteraction(props);
  const branch: EdgeBranch = props.data ?? {};
  const edgeStyle = getEdgeStyle(branch, props.style);
  const labels = getEndpointLabels(props.data);
  return <>
    <EdgePaths geometry={geometry} interaction={interaction} edgeStyle={edgeStyle}
      branchColor={branch.branchColor} markerEnd={props.markerEnd} measurementPathRef={points.measurementPathRef} />
    <EdgeLabels id={props.id} interaction={interaction} branch={branch} labels={labels}
      deletePoint={points.deletePoint} warningPoint={points.warningPoint} />
  </>;
});
CustomEdge.displayName = 'CustomEdge';
