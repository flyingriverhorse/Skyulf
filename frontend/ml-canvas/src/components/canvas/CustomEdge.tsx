import React, { memo, useEffect, useId, useRef, useState } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  EdgeProps,
  Position,
  getSmoothStepPath,
  getStraightPath,
  useReactFlow,
} from '@xyflow/react';
import { X } from 'lucide-react';
import { getReadOnlyMode } from '../../core/hooks/useReadOnlyMode';
import { ConnectionHoverCard } from './ConnectionHoverCard';

export const CustomEdge: React.FC<EdgeProps> = memo(({
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
  selected,
  deletable,
}) => {
  const { deleteElements } = useReactFlow();
  const [hovered, setHovered] = useState(false);
  const [buttonFocused, setButtonFocused] = useState(false);
  const tooltipId = useId();
  const hoverTimer = useRef<ReturnType<typeof setTimeout>>();
  const enter = () => { clearTimeout(hoverTimer.current); setHovered(true); };
  const leave = () => { hoverTimer.current = setTimeout(() => setHovered(false), 150); };
  useEffect(() => () => clearTimeout(hoverTimer.current), []);
  const sourceLabel = typeof data?.sourceLabel === 'string' ? data.sourceLabel : 'Source node';
  const targetLabel = typeof data?.targetLabel === 'string' ? data.targetLabel : 'Target node';
  const showControls = hovered || buttonFocused || selected || data?.isFocused === true;
  const showTooltip = hovered || buttonFocused || data?.isFocused === true;
  const canDelete = deletable && data?.readOnly !== true;

  // When source and target handles are almost collinear along the handle
  // axis (e.g. Right -> Left with nearly identical Y), `getSmoothStepPath`
  // collapses its rounded corners (borderRadius 24) into a degenerate path
  // that renders as a hairline -- or disappears entirely, leaving only the
  // floating × delete button. Detect that case and fall back to a straight
  // path so the connection stays consistently thick and visible.
  const horizontalAxis =
    (sourcePosition === Position.Left || sourcePosition === Position.Right) &&
    (targetPosition === Position.Left || targetPosition === Position.Right);
  const verticalAxis =
    (sourcePosition === Position.Top || sourcePosition === Position.Bottom) &&
    (targetPosition === Position.Top || targetPosition === Position.Bottom);
  const perpendicularOffset = horizontalAxis
    ? Math.abs(targetY - sourceY)
    : verticalAxis
      ? Math.abs(targetX - sourceX)
      : Number.POSITIVE_INFINITY;
  const useStraight = (horizontalAxis || verticalAxis) && perpendicularOffset < 6;

  const [edgePath, labelX, labelY] = useStraight
    ? getStraightPath({ sourceX, sourceY, targetX, targetY })
    : getSmoothStepPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
        borderRadius: 16,
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
    if (!canDelete || getReadOnlyMode()) return;
    // Keep keyboard navigation in the canvas after its delete button disappears.
    evt.currentTarget.closest('.react-flow')?.parentElement?.focus({ preventScroll: true });
    void deleteElements({ edges: [{ id }] });
  };

  return (
    <>
      <g onMouseEnter={enter} onMouseLeave={leave}>
        <BaseEdge
          path={edgePath}
          {...(markerEnd ? { markerEnd } : {})}
          style={{
            ...edgeStyle,
            filter: showControls
              ? `drop-shadow(0 0 4px ${branchColor || 'hsl(var(--primary))'})`
              : edgeStyle.filter,
          }}
          className="react-flow__edge-path"
          interactionWidth={24}
        />
      </g>
      <EdgeLabelRenderer>
        {showTooltip && <ConnectionHoverCard id={tooltipId} x={labelX} y={labelY}
          sourceLabel={sourceLabel} targetLabel={targetLabel} onEnter={enter} onLeave={leave} />}
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
        {canDelete && <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            fontSize: 12,
            pointerEvents: showControls ? 'all' : 'none',
          }}
          className="nodrag nopan"
          onMouseEnter={enter}
          onMouseLeave={leave}
        >
          <button
            type="button"
            className="w-6 h-6 bg-background border border-border text-muted-foreground rounded-full flex items-center justify-center hover:bg-destructive hover:text-destructive-foreground transition-colors shadow-sm focus-ring"
            style={{ opacity: showControls ? 1 : 0 }}
            onFocus={() => setButtonFocused(true)}
            onBlur={() => setButtonFocused(false)}
            onClick={onEdgeClick}
            title="Remove Connection"
            aria-label={`Remove connection from ${sourceLabel} to ${targetLabel}`}
            aria-describedby={showTooltip ? tooltipId : undefined}
          >
            <X size={10} />
          </button>
        </div>}
      </EdgeLabelRenderer>
    </>
  );
});
CustomEdge.displayName = 'CustomEdge';
