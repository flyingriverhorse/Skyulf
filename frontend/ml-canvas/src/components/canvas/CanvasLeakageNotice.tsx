import { AlertTriangle, X } from 'lucide-react';

interface CanvasLeakageNoticeProps {
  message: string;
  onDismiss: () => void;
  onOpenGuide: () => void;
}

/** Retain unlocatable server diagnostics without replacing the preview workspace. */
export function CanvasLeakageNotice({ message, onDismiss, onOpenGuide }: CanvasLeakageNoticeProps) {
  return (
    <section role="alert" aria-label="Leakage safety notice"
      className="absolute top-3 left-14 right-3 z-20 max-w-md rounded-lg border border-red-500/50 bg-background/95 p-3 shadow-md backdrop-blur">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-600 dark:text-red-400" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold text-foreground">Leakage check blocked this run</h2>
          <details className="mt-1 text-xs">
            <summary className="cursor-pointer text-muted-foreground focus-ring">Details</summary>
            <p className="mt-2 max-h-32 overflow-y-auto break-words text-foreground">{message}</p>
          </details>
          <button type="button" onClick={onOpenGuide}
            className="mt-2 rounded text-xs font-medium text-primary underline underline-offset-2 focus-ring">
            Preprocessing guide
          </button>
        </div>
        <button type="button" onClick={onDismiss} aria-label="Dismiss leakage notice"
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded text-muted-foreground hover:bg-secondary focus-ring">
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </section>
  );
}
