import React from 'react';
import { ConnectionLineComponentProps } from 'react-flow-renderer';

const buildSmoothPath = (sx: number, sy: number, tx: number, ty: number) => {
  if (sx === tx && sy === ty) {
    return `M ${sx},${sy}`;
  }

  const horizontalDelta = Math.abs(tx - sx);
  const baseOffset = Math.max(64, Math.min(horizontalDelta * 0.45, 240));
  const controlOffset = Number.isFinite(baseOffset) ? baseOffset : 100;
  const direction = tx >= sx ? 1 : -1;

  const c1x = sx + direction * controlOffset;
  const c1y = sy;
  const c2x = tx - direction * controlOffset;
  const c2y = ty;

  return `M ${sx},${sy} C ${c1x},${c1y} ${c2x},${c2y} ${tx},${ty}`;
};

const ConnectionLine: React.FC<ConnectionLineComponentProps> = ({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
}) => {
  const path = buildSmoothPath(sourceX, sourceY, targetX, targetY);
  const gradientId = 'connection-line-gradient';

  return (
    <g>
      <defs>
        <linearGradient
          id={gradientId}
          gradientUnits="userSpaceOnUse"
          x1={sourceX}
          y1={sourceY}
          x2={targetX}
          y2={targetY}
        >
          <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.8} />
          <stop offset="50%" stopColor="#6366f1" stopOpacity={0.9} />
          <stop offset="100%" stopColor="#a855f7" stopOpacity={0.85} />
        </linearGradient>
        
        {/* Glow filter for connection line */}
        <filter id="connection-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      
      {/* Glow effect base */}
      <path
        d={path}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth={8}
        strokeOpacity={0.2}
        strokeLinecap="round"
        style={{ filter: 'url(#connection-glow)' }}
      />
      
      {/* Main connection line */}
      <path
        d={path}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth={3}
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{
          strokeDasharray: '8 4',
          animation: 'connection-line-dash 1s linear infinite',
        }}
      />
      
      {/* Pulse circle at target */}
      <circle
        cx={targetX}
        cy={targetY}
        r={6}
        fill="#a855f7"
        opacity={0.8}
        style={{
          animation: 'connection-pulse 1.5s ease-in-out infinite',
        }}
      />
    </g>
  );
};

export default ConnectionLine;
