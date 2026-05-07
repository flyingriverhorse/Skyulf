import React from 'react';
import {
  X,
  Merge,
  GitFork,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Gauge,
} from 'lucide-react';

interface CanvasLegendProps {
  onClose: () => void;
}

export const CanvasLegend: React.FC<CanvasLegendProps> = ({ onClose }) => (
  <div className="absolute top-12 left-0 mt-2 w-80 p-3 bg-background border rounded-md shadow-lg text-sm max-h-[80vh] overflow-y-auto z-20">
    <div className="flex items-center justify-between mb-3">
      <h3 className="font-semibold">Canvas Legend</h3>
      <button
        onClick={onClose}
        className="p-1 rounded hover:bg-accent"
        aria-label="Close legend"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>

    <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
      Node Badges
    </div>
    <ul className="space-y-3 mb-4">
      <li className="flex items-start gap-3">
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-400 text-[10px] font-semibold shrink-0 mt-0.5">
          <Merge size={10} />2
        </span>
        <div>
          <div className="font-medium">Safe merge</div>
          <div className="text-xs text-muted-foreground">
            Multiple inputs combined cleanly (no overlapping columns).
          </div>
        </div>
      </li>
      <li className="flex items-start gap-3">
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-600 dark:text-amber-400 ring-1 ring-amber-500/40 text-[10px] font-semibold shrink-0 mt-0.5">
          <Merge size={10} />2
        </span>
        <div>
          <div className="font-medium">Risky merge</div>
          <div className="text-xs text-muted-foreground">
            Inputs share columns &mdash; one branch wins (overwrite). Check Results banner;
            tweak strategy in properties.
          </div>
        </div>
      </li>
      <li className="flex items-start gap-3">
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-500 text-[10px] font-semibold shrink-0 mt-0.5">
          <GitFork size={10} />2
        </span>
        <div>
          <div className="font-medium">Parallel experiments</div>
          <div className="text-xs text-muted-foreground">
            Training/tuning node runs each upstream branch as a separate experiment (no merge).
          </div>
        </div>
      </li>
      <li className="flex items-start gap-3">
        <span className="inline-flex items-center justify-center w-[22px] h-[22px] rounded-full bg-red-500/15 text-red-500 ring-1 ring-red-500/40 shrink-0 mt-0.5">
          <AlertCircle size={10} />
        </span>
        <div>
          <div className="font-medium">Configuration issue</div>
          <div className="text-xs text-muted-foreground">
            Node has missing or invalid settings. Hover the badge for the specific message; open
            the properties panel to fix it.
          </div>
        </div>
      </li>
      <li className="flex items-start gap-3">
        <span className="inline-flex items-center justify-center w-[22px] h-[22px] rounded-full bg-green-50 text-green-700 border border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-900 shrink-0 mt-0.5">
          <CheckCircle2 size={10} />
        </span>
        <div>
          <div className="font-medium">Success</div>
          <div className="text-xs text-muted-foreground">
            Node ran successfully in the last preview / run.
          </div>
        </div>
      </li>
      <li className="flex items-start gap-3">
        <span className="inline-flex items-center justify-center w-[22px] h-[22px] rounded-full bg-red-50 text-red-700 border border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-900 shrink-0 mt-0.5">
          <XCircle size={10} />
        </span>
        <div>
          <div className="font-medium">Failed</div>
          <div className="text-xs text-muted-foreground">
            Node errored. Click it and open the Results panel for the traceback.
          </div>
        </div>
      </li>
    </ul>

    <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
      Performance Overlay
    </div>
    <p className="text-xs text-muted-foreground mb-2">
      Toggle the <Gauge className="w-3 h-3 inline-block align-text-bottom" /> Perf button. Each
      node card grows a colored ring sized by its last-run wall-clock. Thresholds adapt by node
      family — preprocessing runs in milliseconds, single-fit trainers in seconds, and HPO/CV
      tuners legitimately take minutes.
    </p>
    <ul className="space-y-3 mb-4">
      <li className="flex items-start gap-3">
        <span className="inline-block w-5 h-5 rounded-full ring-2 ring-green-500/60 ring-offset-2 ring-offset-background bg-card shrink-0 mt-0.5" />
        <div>
          <div className="font-medium">Fast</div>
          <div className="text-xs text-muted-foreground">
            Preprocess &lt; 500 ms · Trainer &lt; 5 s · Tuner &lt; 1 min. Cheap step, nothing to
            tune.
          </div>
        </div>
      </li>
      <li className="flex items-start gap-3">
        <span className="inline-block w-5 h-5 rounded-full ring-2 ring-amber-500/70 ring-offset-2 ring-offset-background bg-card shrink-0 mt-0.5" />
        <div>
          <div className="font-medium">Medium</div>
          <div className="text-xs text-muted-foreground">
            Preprocess 0.5 – 5 s · Trainer 5 – 60 s · Tuner 1 – 10 min. Healthy range; watch
            growth over time.
          </div>
        </div>
      </li>
      <li className="flex items-start gap-3">
        <span className="inline-block w-5 h-5 rounded-full ring-2 ring-red-500/70 ring-offset-2 ring-offset-background bg-card shrink-0 mt-0.5" />
        <div>
          <div className="font-medium">Slow</div>
          <div className="text-xs text-muted-foreground">
            Preprocess ≥ 5 s · Trainer ≥ 60 s · Tuner ≥ 10 min. Bottleneck candidate — consider
            sampling, caching, fewer trials, or a smaller search space.
          </div>
        </div>
      </li>
    </ul>

    <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
      Edges
    </div>
    <ul className="space-y-3">
      <li className="flex items-start gap-3">
        <svg width="44" height="12" className="shrink-0 mt-1">
          <line x1="0" y1="6" x2="44" y2="6" stroke="#6366f1" strokeWidth="2" strokeDasharray="8 6" />
        </svg>
        <div>
          <div className="font-medium">Standard edge</div>
          <div className="text-xs text-muted-foreground">
            Animated dashed indigo line. Default flow from source to target.
          </div>
        </div>
      </li>
      <li className="flex items-start gap-3">
        <svg width="44" height="12" className="shrink-0 mt-1">
          <line
            x1="0"
            y1="6"
            x2="44"
            y2="6"
            stroke="hsl(0, 80%, 65%)"
            strokeWidth="2"
            strokeDasharray="8 6"
          />
        </svg>
        <div>
          <div className="font-medium">Branch-colored edge</div>
          <div className="text-xs text-muted-foreground">
            Dashed line in a per-branch HSL color (auto-generated, one hue per training/tuning
            terminal). Appears once 2+ training nodes form parallel branches.
          </div>
        </div>
      </li>
      <li className="flex items-start gap-3">
        <svg width="44" height="12" className="shrink-0 mt-1">
          <line
            x1="0"
            y1="6"
            x2="44"
            y2="6"
            stroke="hsl(0, 80%, 65%)"
            strokeWidth="2"
            strokeDasharray="6 4"
            opacity="0.7"
          />
        </svg>
        <div>
          <div className="font-medium">Shared branch edge</div>
          <div className="text-xs text-muted-foreground">
            Same per-branch color but tighter dashes and faded &mdash; this upstream edge feeds
            more than one parallel experiment.
          </div>
        </div>
      </li>
      <li className="flex items-start gap-3">
        <svg width="44" height="12" className="shrink-0 mt-1">
          <line x1="0" y1="6" x2="44" y2="6" stroke="#f59e0b" strokeWidth="4" strokeDasharray="8 6" />
        </svg>
        <div>
          <div className="font-medium">Winning merge edge</div>
          <div className="text-xs text-muted-foreground">
            After a preview run, the branch whose values survived an overlapping-column merge is
            rendered thicker in amber with a &quot;WINS MERGE&quot; label.
          </div>
        </div>
      </li>
    </ul>
  </div>
);
