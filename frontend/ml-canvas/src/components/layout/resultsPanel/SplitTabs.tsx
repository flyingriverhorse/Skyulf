import React from 'react';
import { Layers } from 'lucide-react';
import type { PreviewDataRows } from '../../../core/api/client';

interface SplitTabsProps {
  tabNames: string[];
  datasets: Record<string, PreviewDataRows>;
  totals: Record<string, number>;
  effectiveTab: string | null;
  setActiveTab: (name: string) => void;
}

/** Split-selector tab strip (train / test / X / y …). Shown when the
 *  active dataset (or branch) exposes more than one split. */
export const SplitTabs: React.FC<SplitTabsProps> = ({
  tabNames,
  datasets,
  totals,
  effectiveTab,
  setActiveTab,
}) => (
  <div className="flex items-center border-b bg-muted/5 px-2 pt-2">
    <Layers className="w-3 h-3 text-muted-foreground mr-1 shrink-0" />
    <div className="flex items-center gap-1 overflow-x-auto scrollbar-thin scrollbar-thumb-muted/50">
    {tabNames.map(name => {
      const rows = datasets[name];
      const previewCount = Array.isArray(rows) ? rows.length : 0;
      const total = totals[name] ?? totals._total ?? previewCount;
      const truncated = total > previewCount;
      return (
        <button
          key={name}
          onClick={() => { setActiveTab(name); }}
          className={`shrink-0 flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-t-md border-t border-l border-r transition-colors ${
            effectiveTab === name
              ? 'bg-background text-primary border-b-background translate-y-[1px]'
              : 'bg-muted/30 text-muted-foreground hover:bg-muted/50 border-transparent'
          }`}
          title={
            truncated
              ? `${total} row${total === 1 ? '' : 's'} in ${name} (${previewCount} shown in preview)`
              : `${total} row${total === 1 ? '' : 's'} in ${name}`
          }
        >
          {name}
          <span
            className={`inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-semibold leading-none ${
              effectiveTab === name
                ? 'bg-primary/15 text-primary'
                : 'bg-muted/60 text-muted-foreground'
            }`}
          >
            {total}
          </span>
        </button>
      );
    })}
    </div>
  </div>
);
