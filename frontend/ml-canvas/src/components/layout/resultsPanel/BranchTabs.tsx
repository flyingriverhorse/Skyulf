import React from 'react';
import { GitBranch, AlertTriangle } from 'lucide-react';

interface BranchTabsProps {
  branchLabels: string[];
  activeBranch: string | null;
  setActiveBranch: (label: string) => void;
  branchColors: string[];
  branchAdvisoryCounts: Record<string, number>;
}

/** Branch-selector tab strip shown for multi-branch parallel runs only. */
export const BranchTabs: React.FC<BranchTabsProps> = ({
  branchLabels,
  activeBranch,
  setActiveBranch,
  branchColors,
  branchAdvisoryCounts,
}) => (
  <div className="flex items-center border-b bg-muted/10 px-2 pt-2">
    <GitBranch className="w-3 h-3 text-muted-foreground mr-1 shrink-0" />
    <div className="flex items-center gap-1 overflow-x-auto pb-0 scrollbar-thin scrollbar-thumb-muted/50">
    {branchLabels.map((label, idx) => {
      const isActive = activeBranch === label;
      const advisoryCount = branchAdvisoryCounts[label] ?? 0;
      return (
        <button
          key={label}
          onClick={() => { setActiveBranch(label); }}
          className={`shrink-0 flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-t-md border-t border-l border-r transition-colors ${
            isActive
              ? 'bg-background text-foreground border-b-background translate-y-[1px]'
              : 'bg-muted/30 text-muted-foreground hover:bg-muted/50 border-transparent'
          }`}
          title={
            advisoryCount > 0
              ? `${advisoryCount} merge advisor${advisoryCount === 1 ? 'y' : 'ies'} in this branch`
              : undefined
          }
        >
          <span
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: branchColors[idx] }}
          />
          {label}
          {advisoryCount > 0 && (
            <span
              className="ml-1 inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200 text-[10px] font-semibold leading-none"
              aria-label={`${advisoryCount} merge advisories`}
            >
              <AlertTriangle className="w-2.5 h-2.5" />
              {advisoryCount}
            </span>
          )}
        </button>
      );
    })}
    </div>
  </div>
);
