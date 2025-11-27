import React, { useCallback, useMemo, useState } from 'react';
import { EdgeProps, useReactFlow, Position } from 'react-flow-renderer';
import { getSmartEdge } from '@tisoap/react-flow-smart-edge';

const isFiniteNumber = (value: number) => Number.isFinite(value);

const round = (value: number) => Math.round(value * 100) / 100;

const buildSmoothPath = (sx: number, sy: number, tx: number, ty: number) => {
  if (sx === tx && sy === ty) {
    return `M ${sx},${sy}`;
  }

  const horizontalDelta = Math.abs(tx - sx);
  const verticalDelta = Math.abs(ty - sy);
  
  // Calculate adaptive offset based on both horizontal and vertical distance
  const distance = Math.sqrt(horizontalDelta * horizontalDelta + verticalDelta * verticalDelta);
  const baseOffset = Math.max(80, Math.min(distance * 0.4, 280));
  const controlOffset = Number.isFinite(baseOffset) ? baseOffset : 120;
  
  const direction = tx >= sx ? 1 : -1;
  
  // Add vertical adjustment to create more elegant curves
  const verticalAdjustment = Math.min(Math.abs(ty - sy) * 0.2, 50);

  const c1x = sx + direction * controlOffset;
  const c1y = sy + (ty > sy ? verticalAdjustment : -verticalAdjustment);
  const c2x = tx - direction * controlOffset;
  const c2y = ty - (ty > sy ? verticalAdjustment : -verticalAdjustment);

  return `M ${sx},${sy} C ${c1x},${c1y} ${c2x},${c2y} ${tx},${ty}`;
};

const AnimatedEdge: React.FC<EdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition = Position.Bottom,
  targetPosition = Position.Top,
  markerEnd,
  style,
}) => {
  const { setEdges, getNodes } = useReactFlow();
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseEnter = useCallback(() => {
    setIsHovered(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false);
  }, []);
  
  const coordinates = useMemo(() => {
    if (![sourceX, sourceY, targetX, targetY].every((value) => isFiniteNumber(value))) {
      console.error(`❌ Edge ${id}: Invalid coordinates`, { sourceX, sourceY, targetX, targetY });
      return null;
    }

    return {
      sourceX: round(sourceX),
      sourceY: round(sourceY),
      targetX: round(targetX),
      targetY: round(targetY),
    };
  }, [sourceX, sourceY, targetX, targetY, id]);

  const edgeParams = useMemo(() => {
    if (!coordinates) {
      return null;
    }

    const { sourceX, sourceY, targetX, targetY } = coordinates;
    
    // 1. Try Smart Edge (Obstacle Avoidance)
    try {
      // Ensure nodes have dimensions so they are treated as obstacles
      const nodes = getNodes().map(n => ({ 
        ...n, 
        width: n.width && n.width > 0 ? n.width : 320, 
        height: n.height && n.height > 0 ? n.height : 170 
      }));

      const smartResult = getSmartEdge({
        sourcePosition,
        targetPosition,
        sourceX,
        sourceY,
        targetX,
        targetY,
        nodes,
        options: {
          nodePadding: 40, // Increased padding for better separation
          gridRatio: 10,
        },
      });

      if (smartResult && 'svgPathString' in smartResult) {
        const { svgPathString, edgeCenterX, edgeCenterY } = smartResult;
        return { path: svgPathString, labelX: edgeCenterX, labelY: edgeCenterY };
      }
    } catch (e) {
      // Ignore smart edge errors and fall back
    }

    // 2. Fallback to Original Smooth Path
    const path = buildSmoothPath(sourceX, sourceY, targetX, targetY);
    const labelX = (sourceX + targetX) / 2;
    const labelY = (sourceY + targetY) / 2;
    
    return { path, labelX, labelY };
  }, [coordinates, sourcePosition, targetPosition, getNodes]);

  const gradientId = useMemo(() => `animated-edge-gradient-${id}`, [id]);

  if (!coordinates || !edgeParams) {
    return null;
  }

  const { sourceX: sx, sourceY: sy, targetX: tx, targetY: ty } = coordinates;
  const { path: edgePath, labelX, labelY } = edgeParams;
  
  const removeButtonSize = 32;
  const foreignObjectX = labelX - removeButtonSize / 2;
  const foreignObjectY = labelY - removeButtonSize / 2;
  const edgeClassName = `animated-edge${isHovered ? ' animated-edge--hovered' : ''}`;
  const removeButtonClassName = `canvas-edge-remove${isHovered ? ' canvas-edge-remove--visible' : ''}`;

  return (
    <g
      className={edgeClassName}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <defs>
        <linearGradient
          id={gradientId}
          gradientUnits="userSpaceOnUse"
          x1={sx}
          y1={sy}
          x2={tx}
          y2={ty}
        >
          <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.85} />
          <stop offset="50%" stopColor="#6366f1" stopOpacity={0.95} />
          <stop offset="100%" stopColor="#a855f7" stopOpacity={0.9} />
        </linearGradient>
        
        {/* Glow filter for enhanced visual effect */}
        <filter id={`edge-glow-${id}`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Invisible interaction layer for better hover detection */}
      <path 
        d={edgePath} 
        fill="none" 
        stroke="transparent"
        strokeWidth={20}
        style={{ cursor: 'pointer' }}
        className="animated-edge__interaction" 
      />

      {/* Glow/shadow base layer */}
      <path
        d={edgePath}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth={6}
        strokeOpacity={0.2}
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ filter: `url(#edge-glow-${id})` }}
      />

      {/* Main edge path */}
      <path
        d={edgePath}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth={3}
        strokeLinecap="round"
        strokeLinejoin="round"
        markerEnd={markerEnd}
        className="animated-edge__base"
        style={{
          ...style,
          transition: 'stroke-width 0.2s ease',
        }}
      />

      {/* Animated dash trail */}
      <path
        d={edgePath}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth={isHovered ? 3.5 : 3}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeDasharray="16 12"
        className="animated-edge__trail"
        style={{
          transition: 'stroke-width 0.2s ease',
        }}
      />

      {/* Sparkle effect */}
      <path
        d={edgePath}
        fill="none"
        stroke="rgba(248, 250, 252, 0.4)"
        strokeWidth={1.5}
        strokeDasharray="3 8"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="animated-edge__spark"
      />

      <foreignObject
        className={removeButtonClassName}
        x={foreignObjectX}
        y={foreignObjectY}
        width={removeButtonSize}
        height={removeButtonSize}
        requiredExtensions="http://www.w3.org/1999/xhtml"
      >
        <div className="canvas-edge-remove__wrapper">
          <button
            type="button"
            className="canvas-edge-remove__button"
            onClick={(event) => {
              event.stopPropagation();
              setEdges((edges) => edges.filter((edge) => edge.id !== id));
            }}
            aria-label="Remove connection"
          >
            ×
          </button>
        </div>
      </foreignObject>
    </g>
  );
};

export default AnimatedEdge;