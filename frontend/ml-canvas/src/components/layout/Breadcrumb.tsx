import React from 'react';
import { Database, GitBranch, BarChart2, Rocket, ChevronRight, type LucideIcon } from 'lucide-react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';

/**
 * Breadcrumb context strip for the Navbar (M6).
 *
 * Shows the user's current orientation across views: which dataset is
 * active, and what view they're looking at. The dataset segment doubles
 * as a click-back to the canvas — useful when the user has wandered
 * into Experiments / Inference and wants to jump back to the editor.
 *
 * Intentionally compact: one row, hidden on small viewports where the
 * Navbar's primary segmented control already fills the bar.
 */
const VIEW_META: Record<
  'canvas' | 'experiments' | 'inference',
  { label: string; Icon: LucideIcon }
> = {
  canvas: { label: 'Canvas', Icon: GitBranch },
  experiments: { label: 'Experiments', Icon: BarChart2 },
  inference: { label: 'Inference', Icon: Rocket },
};

export const Breadcrumb: React.FC = () => {
  const nodes = useGraphStore((s) => s.nodes);
  const activeView = useViewStore((s) => s.activeView);
  const setView = useViewStore((s) => s.setView);

  const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
  const datasetName = (datasetNode?.data.datasetName as string | undefined) ?? null;

  // Without a dataset there's nothing useful to anchor on — let the
  // primary navigation carry the orientation work.
  if (!datasetName) return null;

  const view = VIEW_META[activeView];
  const ViewIcon = view.Icon;

  return (
    <nav
      aria-label="Breadcrumb"
      className="hidden md:flex items-center gap-1.5 text-xs text-muted-foreground absolute left-4 top-1/2 -translate-y-1/2 max-w-[40%]"
    >
      <button
        onClick={() => setView('canvas')}
        className="flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-secondary/60 hover:text-foreground transition-colors min-w-0"
        title={`Dataset: ${datasetName} — click to open Canvas`}
        data-testid="breadcrumb-dataset"
      >
        <Database className="w-3 h-3 flex-shrink-0" />
        <span className="truncate font-medium">{datasetName}</span>
      </button>
      <ChevronRight className="w-3 h-3 flex-shrink-0" aria-hidden="true" />
      <span
        className="flex items-center gap-1 px-1.5 py-0.5 text-foreground"
        aria-current="page"
        data-testid="breadcrumb-view"
      >
        <ViewIcon className="w-3 h-3 flex-shrink-0" />
        <span>{view.label}</span>
      </span>
    </nav>
  );
};
