import React from 'react';
import { BarChart2, GitBranch, Eye, Pencil, Rocket } from 'lucide-react';
import { useViewStore } from '../../core/store/useViewStore';
import { useReadOnlyMode } from '../../core/hooks/useReadOnlyMode';
import { useViewport } from '../../core/hooks/useViewport';

export const Navbar: React.FC = () => {
  const { activeView, setView } = useViewStore();
  const setReadOnlyOverride = useViewStore((s) => s.setReadOnlyOverride);
  const readOnly = useReadOnlyMode();
  const { isTablet } = useViewport();

  // Show the read-only chip only on canvas view. Tablet users get an
  // info chip explaining why edit tools are hidden; desktop users only
  // see it when they've opted in via the toggle.
  const showReadOnlyChip = activeView === 'canvas' && (readOnly || isTablet);

  const toggleReadOnly = (): void => {
    // From the user's POV: clicking flips read-only on/off and "pins"
    // the choice (override leaves `auto`). We resolve `auto` -> the
    // current effective value, then flip.
    setReadOnlyOverride(readOnly ? 'off' : 'on');
  };

  return (
    <div className="h-14 border-b bg-card px-4 flex items-center justify-center shrink-0 relative">
      {/* Navigation */}
      <div className="flex items-center gap-1 bg-secondary/50 p-1 rounded-lg">
        <button
          onClick={() => setView('canvas')}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
            activeView === 'canvas' 
              ? 'bg-background shadow-sm text-foreground' 
              : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
          }`}
        >
          <GitBranch className="w-4 h-4" />
          Canvas
        </button>
        <button
          onClick={() => setView('experiments')}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
            activeView === 'experiments' 
              ? 'bg-background shadow-sm text-foreground' 
              : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
          }`}
        >
          <BarChart2 className="w-4 h-4" />
          Experiments
        </button>
        <button
          onClick={() => setView('inference')}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
            activeView === 'inference' 
              ? 'bg-background shadow-sm text-foreground' 
              : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
          }`}
        >
          <Rocket className="w-4 h-4" />
          Inference
        </button>
      </div>

      {showReadOnlyChip && (
        <button
          onClick={toggleReadOnly}
          title={
            readOnly
              ? 'Read-only canvas (tablet view). Click to enable editing.'
              : 'Editing enabled. Click to switch to read-only.'
          }
          className={`absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors border ${
            readOnly
              ? 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30 hover:bg-amber-500/20'
              : 'bg-secondary/50 text-muted-foreground border-transparent hover:bg-secondary'
          }`}
          aria-pressed={readOnly}
        >
          {readOnly ? <Eye className="w-3.5 h-3.5" /> : <Pencil className="w-3.5 h-3.5" />}
          <span className="hidden sm:inline">{readOnly ? 'Read-only' : 'Editing'}</span>
        </button>
      )}
    </div>
  );
};
