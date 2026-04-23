import React from 'react';
import { 
  BaseEdge, 
  EdgeLabelRenderer, 
  EdgeProps, 
  getSmoothStepPath,
  useReactFlow
} from '@xyflow/react';
import { X } from 'lucide-react';

export const CustomEdge: React.FC<EdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  data,
}) => {
  const { deleteElements } = useReactFlow();
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 24,
  });

  const branchColor = (data as Record<string, unknown>)?.branchColor as string | undefined;
  const branchLabel = (data as Record<string, unknown>)?.branchLabel as string | undefined;
  const branchShared = (data as Record<string, unknown>)?.branchShared as boolean | undefined;
  const isMergeWinner = (data as Record<string, unknown>)?.isMergeWinner as boolean | undefined;
  const edgeStyle = branchColor
    ? {
        ...style,
        stroke: branchColor,
        strokeDasharray: branchShared ? '6 4' : undefined,  // dashed = feeds multiple experiments
        filter: undefined,
        strokeWidth: isMergeWinner ? 4 : 2,
        opacity: branchShared ? 0.7 : 1,
      }
    : {
        ...style,
        strokeWidth: isMergeWinner ? 4 : 2,
        stroke: isMergeWinner ? '#f59e0b' : style.stroke,
        filter: isMergeWinner ? undefined : style.filter,
      };

  const onEdgeClick = (evt: React.MouseEvent) => {
    evt.stopPropagation();
    deleteElements({ edges: [{ id }] });
  };

  return (
    <>
      {/* Invisible wider path for easier selection */}
      <BaseEdge 
        path={edgePath} 
        style={{ strokeWidth: 20, stroke: 'transparent', cursor: 'pointer' }} 
      />
      {/* Visible path */}
      <BaseEdge 
        path={edgePath} 
        markerEnd={markerEnd} 
        style={edgeStyle} 
        className="react-flow__edge-path"
      />
      <EdgeLabelRenderer>
        {isMergeWinner && (
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -100%) translate(${labelX}px,${labelY - (branchLabel ? 40 : 16)}px)`,
              fontSize: 9,
              fontWeight: 700,
              letterSpacing: '0.06em',
              color: '#f59e0b',
              backgroundColor: 'hsl(var(--background) / 0.95)',
              border: '1px solid #f59e0b80',
              borderRadius: 4,
              padding: '1px 6px',
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
              lineHeight: '14px',
              textTransform: 'uppercase',
            }}
            title="This branch won the merge tiebreak on overlapping columns."
          >
            Wins merge
          </div>
        )}
        {branchLabel && branchColor && (
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -100%) translate(${labelX}px,${labelY - 16}px)`,
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: '0.02em',
              color: branchColor,
              backgroundColor: 'hsl(var(--background) / 0.9)',
              border: `1px solid ${branchColor}50`,
              borderRadius: 6,
              padding: '2px 8px',
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
              lineHeight: '16px',
              boxShadow: `0 0 6px ${branchColor}30`,
            }}
          >
            {branchLabel}
          </div>
        )}
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            fontSize: 12,
            pointerEvents: 'all',
          }}
          className="nodrag nopan"
        >
          <button
            className="w-5 h-5 bg-background border border-border text-muted-foreground rounded-full flex items-center justify-center hover:bg-destructive hover:text-destructive-foreground transition-colors shadow-sm"
            onClick={onEdgeClick}
            title="Remove Connection"
          >
            <X size={10} />
          </button>
        </div>
      </EdgeLabelRenderer>
    </>
  );
};
