import React from 'react';
import { BarChart2, GitBranch, Rocket } from 'lucide-react';
import { useViewStore } from '../../core/store/useViewStore';

export const Navbar: React.FC = () => {
  const { activeView, setView } = useViewStore();

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
    </div>
  );
};
