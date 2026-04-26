#!/usr/bin/env node
// Phase C #28 — Bundle size budget enforcer.
// Zero-dependency: uses Node's built-in `zlib` for gzip and `fs` for I/O.
// Fails (exit 1) when any tracked chunk exceeds its gzip ceiling so a
// careless re-import (e.g. swapping the slim plotly bundle back to the
// full one) breaks CI before it ships.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
// Vite is configured (vite.config.ts) to emit into the repo's static/
// directory, not a local dist/. Resolve relative to that.
const distAssets = join(__dirname, '..', '..', '..', 'static', 'ml_canvas', 'assets');

// Per-chunk gzip ceilings in bytes. Bump deliberately when a feature
// genuinely needs the headroom; the point is to make the bump visible
// in code review.
// NOTE: vendor-plotly is currently the full plotly.js-dist-min build
// (~2.85 MB gzip). When item #14 (slim plotly-gl3d swap) is reapplied
// this drops to ~542 KB and the budget should be ratcheted down.
//
// `kind: 'vendor'` is required to be present (missing chunk = WARN).
// `kind: 'route'` is optional — the chunk only exists once the route
// is built; missing it is silently skipped so adding/removing routes
// doesn't fight the budget gate.
const BUDGETS = [
  // Vendor chunks (always emitted, manualChunks in vite.config.ts)
  { prefix: 'vendor-plotly', maxGzipBytes: 3100 * 1024, label: 'vendor-plotly', kind: 'vendor' },
  { prefix: 'vendor-charts', maxGzipBytes: 220 * 1024,  label: 'vendor-charts', kind: 'vendor' },
  { prefix: 'vendor-flow',   maxGzipBytes: 80 * 1024,   label: 'vendor-flow',   kind: 'vendor' },
  { prefix: 'vendor-react',  maxGzipBytes: 70 * 1024,   label: 'vendor-react',  kind: 'vendor' },
  { prefix: 'vendor-utils',  maxGzipBytes: 90 * 1024,   label: 'vendor-utils',  kind: 'vendor' },
  // App entry
  { prefix: 'index',         maxGzipBytes: 200 * 1024,  label: 'index (main)',  kind: 'vendor' },
  // Lazy route chunks — keep tight so an EDA-only regression surfaces
  // here rather than getting absorbed by the global index ceiling.
  { prefix: 'EDAPage',       maxGzipBytes: 140 * 1024,  label: 'route:EDA',         kind: 'route' },
  { prefix: 'DataDriftPage', maxGzipBytes: 20 * 1024,   label: 'route:DataDrift',   kind: 'route' },
  { prefix: 'ModelRegistry', maxGzipBytes: 15 * 1024,   label: 'route:ModelRegistry', kind: 'route' },
  { prefix: 'DeploymentsPage', maxGzipBytes: 10 * 1024, label: 'route:Deployments', kind: 'route' },
];

function listAssets() {
  try {
    return readdirSync(distAssets).filter((f) => f.endsWith('.js'));
  } catch (err) {
    console.error(`[size-check] cannot read ${distAssets}: ${err.message}`);
    console.error('[size-check] run `npm run build` first.');
    process.exit(2);
  }
}

function fmt(n) {
  return (n / 1024).toFixed(1) + ' KB';
}

const files = listAssets();
let failed = 0;

for (const budget of BUDGETS) {
  const match = files.find((f) => f.startsWith(budget.prefix));
  if (!match) {
    if (budget.kind === 'route') {
      // Route chunks are optional — only flag at INFO level.
      console.log(`[size-check] SKIP  ${budget.label.padEnd(22)} (chunk not emitted)`);
    } else {
      console.warn(`[size-check] WARN  ${budget.label.padEnd(22)} no chunk starting with "${budget.prefix}"`);
    }
    continue;
  }
  const path = join(distAssets, match);
  const raw = readFileSync(path);
  const gz = gzipSync(raw).length;
  const pct = ((gz / budget.maxGzipBytes) * 100).toFixed(0);
  const status = gz > budget.maxGzipBytes ? 'FAIL' : 'OK';
  console.log(
    `[size-check] ${status.padEnd(4)} ${budget.label.padEnd(22)} ` +
      `raw=${fmt(statSync(path).size).padStart(10)}  ` +
      `gz=${fmt(gz).padStart(10)}  ` +
      `budget=${fmt(budget.maxGzipBytes).padStart(10)}  (${pct}%)`,
  );
  if (gz > budget.maxGzipBytes) failed += 1;
}

if (failed > 0) {
  console.error(`[size-check] ${failed} chunk(s) over budget.`);
  process.exit(1);
}
console.log('[size-check] all chunks within budget.');
