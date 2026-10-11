import { useEffect, useId, useState } from 'react';

// Geometry and timing from the supplied transparent logo scene; the export editor is not needed.
const DURATION = 7.6;
const ORBIT_START = 1.6;
const SETTLE = 3.8;
const PLANET = { x: 607, y: 683, r: 254 };
const MOON_RADIUS = 43;
const ORBIT_ANGLE = -32 * Math.PI / 180;
const PHASE = 24;
const COLOR_STOPS: [number, string][] = [
  [0, '#ffbf3a'], [90, '#ffcf5c'], [140, '#ffe39a'], [165, '#eaf6ff'],
  [185, '#7dbdff'], [215, '#2a6cf0'], [260, '#1546c9'], [285, '#3b6fe0'],
  [310, '#ffcf6a'], [360, '#ffbf3a'],
];

const outCubic = (value: number) => 1 + (value - 1) ** 3;
const inOutCubic = (value: number) => value < 0.5 ? 4 * value ** 3 : 1 + 4 * (value - 1) ** 3;
const outBack = (value: number) => 1 + 2.70158 * (value - 1) ** 3 + 1.70158 * (value - 1) ** 2;

/** Apply the supplied scene's clamped interpolation and easing. */
function motion(time: number, start: number, end: number, from = 0, to = 1, ease = outCubic) {
  const progress = Math.max(0, Math.min(1, (time - start) / (end - start)));
  return from + (to - from) * ease(progress);
}

/** Stop scheduling frames as soon as the loading scene unmounts. */
function useSceneTime() {
  const [time, setTime] = useState(0);
  useEffect(() => {
    let start: number | undefined;
    let frame: number;
    const tick = (now: number) => {
      start ??= now;
      setTime(((now - start) / 1000) % DURATION);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);
  return time;
}

/** Locate the moon and depth on the original tilted ellipse. */
function orbitPoint(degrees: number) {
  const radians = degrees * Math.PI / 180;
  const horizontal = 530 * Math.cos(radians);
  const vertical = 184 * Math.sin(radians);
  return {
    x: 625 + horizontal * Math.cos(ORBIT_ANGLE) - vertical * Math.sin(ORBIT_ANGLE),
    y: 622 + horizontal * Math.sin(ORBIT_ANGLE) + vertical * Math.cos(ORBIT_ANGLE),
    depth: Math.sin(radians),
  };
}

const channels = (hex: string) => [1, 3, 5].map(index => Number.parseInt(hex.slice(index, index + 2), 16));

/** Interpolate the original blue and gold orbit color stops. */
function orbitColor(degrees: number) {
  const angle = ((degrees % 360) + 360) % 360;
  for (let index = 0; index < COLOR_STOPS.length - 1; index++) {
    const [start, startColor] = COLOR_STOPS[index]!;
    const [end, endColor] = COLOR_STOPS[index + 1]!;
    if (angle <= end) {
      const progress = (angle - start) / (end - start);
      const target = channels(endColor);
      return `rgb(${channels(startColor).map((value, channel) => Math.round(value + (target[channel]! - value) * progress)).join(',')})`;
    }
  }
  return COLOR_STOPS[0]![1];
}

interface Segment {
  key: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  width: number;
}

/** Draw the original 120 segments with the planet occluding the rear half. */
function ringSegments(progress: number) {
  const front: Segment[] = [];
  const back: Segment[] = [];
  const end = PHASE + 360 * progress;
  for (let key = 0; key < 120; key++) {
    const start = PHASE + key * 3;
    if (start >= end) break;
    const stop = Math.min(start + 3, end);
    const middle = (start + stop) / 2;
    const first = orbitPoint(start);
    const last = orbitPoint(stop);
    const depth = Math.sin(middle * Math.PI / 180);
    const segment = { key, x1: first.x, y1: first.y, x2: last.x, y2: last.y, color: orbitColor(middle), width: 9 + 3.5 * depth };
    (depth < 0 ? back : front).push(segment);
  }
  return { front, back };
}

function Ring({ segments, glow, opacity, id }: { segments: Segment[]; glow: number; opacity: number; id: string }) {
  return (
    <g opacity={opacity}>
      <g filter={`url(#${id}-blur)`} opacity={glow}>
        {segments.map(segment => <line key={segment.key} x1={segment.x1} y1={segment.y1} x2={segment.x2} y2={segment.y2} stroke={segment.color} strokeWidth={segment.width * 2.8} strokeLinecap="round" />)}
      </g>
      {segments.map(segment => <line key={segment.key} x1={segment.x1} y1={segment.y1} x2={segment.x2} y2={segment.y2} stroke={segment.color} strokeWidth={segment.width} strokeLinecap="round" />)}
    </g>
  );
}

function Moon({ point, scale, glow, id }: { point: ReturnType<typeof orbitPoint>; scale: number; glow: number; id: string }) {
  const depth = point.depth < 0 ? 0.86 + 0.14 * (1 + point.depth) : 1;
  const radius = MOON_RADIUS * scale * depth;
  if (radius <= 0.1) return null;
  return (
    <g>
      <circle cx={point.x} cy={point.y} r={radius * 1.7} fill="#ffc94a" opacity={0.45 * glow} filter={`url(#${id}-blur)`} />
      <circle cx={point.x} cy={point.y} r={radius} fill={`url(#${id}-moon)`} />
    </g>
  );
}

function Planet({ time, id }: { time: number; id: string }) {
  const scale = motion(time, 0.1, 1.25, 0.35, 1, outBack);
  const opacity = motion(time, 0, 0.6);
  const halo = motion(time, 0.3, 1.6);
  return (
    <g opacity={opacity} transform={`translate(${PLANET.x} ${PLANET.y}) scale(${scale}) translate(${-PLANET.x} ${-PLANET.y})`}>
      <circle cx={PLANET.x} cy={PLANET.y} r={PLANET.r * 1.18} fill="#3d8cff" opacity={0.22 * halo} filter={`url(#${id}-blur-wide)`} />
      <circle cx={PLANET.x} cy={PLANET.y} r={PLANET.r + 4} fill="none" stroke="#5ab4ff" strokeWidth="16" opacity={0.55 * halo} filter={`url(#${id}-blur)`} />
      <circle cx={PLANET.x} cy={PLANET.y} r={PLANET.r} fill={`url(#${id}-planet)`} />
    </g>
  );
}

/** Play the supplied SVG scene on the page's own background. */
export function LoadingLogoAnimation({ logoUrl }: { logoUrl: string }) {
  const time = useSceneTime();
  const id = useId();
  const lap = motion(time, ORBIT_START + 0.25, SETTLE, 0, 1, inOutCubic);
  const moonScale = motion(time, ORBIT_START, ORBIT_START + 0.45, 0, 1, outBack);
  const moon = orbitPoint(PHASE + 360 * lap);
  const { front, back } = ringSegments(lap);
  const arrival = time <= SETTLE - 0.05 || time >= SETTLE + 0.8 ? 0 : Math.sin(Math.PI * (time - SETTLE + 0.05) / 0.85);
  const glow = 0.32 + 0.45 * arrival;
  const burstRadius = motion(time, SETTLE - 0.05, SETTLE + 0.9, MOON_RADIUS, MOON_RADIUS * 4.2);
  const burstOpacity = time < SETTLE - 0.05 ? 0 : 0.7 * (1 - motion(time, SETTLE - 0.05, SETTLE + 0.9));
  const toImage = motion(time, SETTLE + 0.35, SETTLE + 1.2);
  const camera = 1 + 0.05 * time / DURATION;
  const fade = 1 - motion(time, DURATION - 0.6, DURATION);

  return (
    <svg width={64} height={64} viewBox="420 0 1080 1080" className="h-16 w-16" aria-hidden="true">
      <defs>
        <radialGradient id={`${id}-planet`} cx="0.44" cy="0.5" r="0.56" fx="0.4" fy="0.45">
          <stop offset="0" stopColor="#0a3594" />
          <stop offset="0.55" stopColor="#0b42ad" />
          <stop offset="0.8" stopColor="#1557c9" />
          <stop offset="0.93" stopColor="#2f86ec" />
          <stop offset="1" stopColor="#8fd6ff" />
        </radialGradient>
        <radialGradient id={`${id}-moon`} cx="0.38" cy="0.35" r="0.7">
          <stop offset="0" stopColor="#fff6cf" />
          <stop offset="0.45" stopColor="#ffd25a" />
          <stop offset="1" stopColor="#f2a51c" />
        </radialGradient>
        <filter id={`${id}-blur`} x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="10" /></filter>
        <filter id={`${id}-blur-wide`} x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="34" /></filter>
      </defs>
      <g opacity={fade} transform={`translate(960 540) scale(${camera}) translate(-960 -540)`}>
        <g transform="translate(960 540) scale(0.86) translate(-627 -627)">
          <g opacity={1 - toImage}>
            <Ring segments={back} glow={glow} opacity={0.92} id={id} />
            {moon.depth < 0 && <Moon point={moon} scale={moonScale} glow={1} id={id} />}
            <Planet time={time} id={id} />
            <Ring segments={front} glow={glow} opacity={1} id={id} />
            {moon.depth >= 0 && <Moon point={moon} scale={moonScale * (1 + 0.22 * arrival)} glow={1 + arrival} id={id} />}
          </g>
          <image href={logoUrl} x="0" y="0" width="1254" height="1254" opacity={toImage} />
          <circle cx={moon.x} cy={moon.y} r={burstRadius} fill="none" stroke="#ffd35e" strokeWidth="6" opacity={burstOpacity} />
        </g>
      </g>
    </svg>
  );
}
