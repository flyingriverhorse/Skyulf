import React from 'react';
import { ChevronDown, ChevronRight, AlertTriangle, Wand2 } from 'lucide-react';
import type { MergeWarning } from '../../../core/api/client';
import type { ConfirmOptions } from '../../shared/ConfirmDialog';
import { toast } from '../../../core/toast';

interface MergeWarningsBannerProps {
  mergeWarnings: MergeWarning[];
  mergeWarningsOpen: boolean;
  setMergeWarningsOpen: React.Dispatch<React.SetStateAction<boolean>>;
  nodeLabelMap: Record<string, string>;
  confirm: (options: ConfirmOptions) => Promise<boolean>;
  chainSiblings: (consumerId: string, orderedInputIds: string[]) => boolean;
}

/** Collapsed-by-default banner for engine-emitted merge advisories (sibling
 *  fan-in etc.). Click to expand full per-warning detail (inputs, overlap
 *  columns, winner). */
export const MergeWarningsBanner: React.FC<MergeWarningsBannerProps> = ({
  mergeWarnings,
  mergeWarningsOpen,
  setMergeWarningsOpen,
  nodeLabelMap,
  confirm,
  chainSiblings,
}) => (
  <div className="bg-amber-50 dark:bg-amber-950/20 border-b dark:border-amber-900/30">
    <button
      type="button"
      onClick={() => setMergeWarningsOpen((v) => !v)}
      className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-amber-900 dark:text-amber-200 hover:bg-amber-100/60 dark:hover:bg-amber-900/30 transition-colors"
      aria-expanded={mergeWarningsOpen}
    >
      {mergeWarningsOpen ? (
        <ChevronDown className="w-3.5 h-3.5 flex-shrink-0" />
      ) : (
        <ChevronRight className="w-3.5 h-3.5 flex-shrink-0" />
      )}
      <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
      <span className="font-medium">
        {mergeWarnings.length} merge advisor{mergeWarnings.length === 1 ? 'y' : 'ies'}
      </span>
      <span className="opacity-80 truncate">
        — {mergeWarnings
          .map((w) => nodeLabelMap[w.node_id] ?? w.node_id)
          .slice(0, 3)
          .join(', ')}
        {mergeWarnings.length > 3 ? `, +${mergeWarnings.length - 3} more` : ''}
      </span>
    </button>
    {mergeWarningsOpen && (
      <div className="px-2 pb-2 space-y-1.5">
        {mergeWarnings.map((w, idx) => {
          // Row-wise merge dropped non-shared columns: render a
          // simpler advisory (no inputs / overlap to show).
          if (w.kind === 'row_concat_drop') {
            const dropped = w.dropped_columns ?? [];
            return (
              <div key={idx} className="flex items-start gap-2 text-xs text-amber-900 dark:text-amber-200 pl-5">
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <span className="font-medium">
                    {nodeLabelMap[w.node_id] ?? w.node_id}
                  </span>{' '}
                  row-wise merge{w.part ? ` (${w.part})` : ''} dropped{' '}
                  {dropped.length} non-shared column
                  {dropped.length === 1 ? '' : 's'}:{' '}
                  <span className="font-mono bg-amber-100 dark:bg-amber-900/40 px-1 rounded">
                    {dropped.slice(0, 6).join(', ')}
                    {dropped.length > 6 ? `, +${dropped.length - 6} more` : ''}
                  </span>
                  . Only columns present in every input are kept when row counts differ.
                </div>
              </div>
            );
          }
          const inputs = w.inputs ?? [];
          const inputLabels = inputs.map((i) => nodeLabelMap[i] ?? i);
          const winner = w.winner_input
            ? (nodeLabelMap[w.winner_input] ?? w.winner_input)
            : inputLabels[inputLabels.length - 1];
          const overlap = w.overlap_columns ?? [];
          const canAutoChain = overlap.length > 0 && inputs.length >= 2;
          return (
            <div key={idx} className="flex items-start gap-2 text-xs text-amber-900 dark:text-amber-200 pl-5">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <span className="font-medium">
                  {nodeLabelMap[w.node_id] ?? w.node_id}
                </span>{' '}
                merges {inputs.length} parallel branches:{' '}
                {inputLabels.map((label, i) => (
                  <span key={i}>
                    <span className="font-mono bg-amber-100 dark:bg-amber-900/40 px-1 rounded">
                      {label}
                    </span>
                    {i < inputLabels.length - 1 && ' + '}
                  </span>
                ))}
                .{' '}
                {overlap.length > 0 ? (
                  <>
                    {overlap.length} overlapping column
                    {overlap.length === 1 ? '' : 's'}{' '}
                    (<span className="font-mono">
                      {overlap.slice(0, 4).join(', ')}
                      {overlap.length > 4 ? `, +${overlap.length - 4} more` : ''}
                    </span>){' '}
                    take values from{' '}
                    <span className="font-mono font-semibold">{winner}</span>;
                    unique columns from the others are kept as-is.
                  </>
                ) : (
                  <>
                    No column overlap — all columns from all branches are kept.
                  </>
                )}{' '}
                For sequential application, chain them instead.
                {canAutoChain && (
                  <div className="mt-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        const consumerLabel = nodeLabelMap[w.node_id] ?? w.node_id;
                        const chainStr = [...inputLabels, consumerLabel].join(' → ');
                        void (async () => {
                          const ok = await confirm({
                            title: 'Rewire as a linear chain?',
                            message: (
                              <span>
                                {chainStr}
                                <br />
                                <br />
                                <span className="text-xs text-slate-500">Use Ctrl+Z to undo.</span>
                              </span>
                            ),
                            confirmLabel: 'Rewire',
                          });
                          if (!ok) return;
                          const success = chainSiblings(w.node_id, inputs);
                          if (!success) {
                            toast.error('Auto-chain failed', 'Re-run preview and try again.');
                          }
                        })();
                      }}
                      className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-medium rounded border border-amber-400 dark:border-amber-600 bg-amber-100 dark:bg-amber-900/40 hover:bg-amber-200 dark:hover:bg-amber-900/60 text-amber-900 dark:text-amber-100 transition-colors"
                      title="Rewire fan-in into a linear chain (no fan-in, no overwrite)"
                    >
                      <Wand2 className="w-3 h-3" />
                      Chain instead
                    </button>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    )}
  </div>
);
