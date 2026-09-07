import { ModalShell } from '../../shared/ModalShell';
import type { RunControls } from './_hooks/useRunControls';

/** Review model scope and blocking issues before using the existing experiment handler. */
export function ExperimentRunDialog({ isOpen, models, blockReason, onClose, onSubmit, onReviewIssues }: {
  isOpen: boolean;
  models: RunControls['experimentModels'];
  blockReason: string;
  onClose: () => void;
  onSubmit: () => void;
  onReviewIssues: () => void;
}) {
  return <ModalShell isOpen={isOpen} onClose={onClose} title="Run all experiments?" size="lg"
    footer={<div className="flex flex-wrap justify-end gap-2">
      <button type="button" onClick={onClose} className="action-secondary rounded-md px-3 py-2 text-sm focus-ring">Cancel</button>
      <button type="button" onClick={onSubmit} disabled={Boolean(blockReason)} aria-describedby={blockReason ? 'experiment-run-blocked' : 'experiment-run-help'}
        className="action-primary rounded-md px-3 py-2 text-sm disabled:opacity-50 focus-ring">Queue experiments</button>
    </div>}>
    <div className="space-y-4 p-4 text-sm">
      <p id="experiment-run-help" className="text-muted-foreground">Runs training or tuning for the models below in the background. Parallel inputs may create multiple experiments. Use Preview data to inspect preprocessing separately.</p>
      <ul className="divide-y rounded-md border">
        {models.map(model => <li key={model.id} className="flex min-w-0 items-start justify-between gap-3 p-3">
          <div className="min-w-0 [overflow-wrap:anywhere]"><p className="font-medium">{model.name}</p><p className="text-xs text-muted-foreground">{model.model}</p></div>
          <span className="shrink-0 rounded bg-muted px-2 py-1 text-xs">{model.action}</span>
        </li>)}
      </ul>
      {blockReason && <div className="space-y-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3">
        <p id="experiment-run-blocked" className="break-words">{blockReason}</p>
        <button type="button" onClick={onReviewIssues} className="rounded text-primary underline underline-offset-2 focus-ring">Review validation issues</button>
      </div>}
    </div>
  </ModalShell>;
}
