import React, { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { ModalShell } from './ModalShell';

export type ConfirmVariant = 'default' | 'danger';

export interface ConfirmOptions {
  title: string;
  message: React.ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: ConfirmVariant;
}

type Resolver = (ok: boolean) => void;

interface ConfirmContextValue {
  /** Returns a promise that resolves `true` on confirm, `false` on cancel/dismiss. */
  confirm: (options: ConfirmOptions) => Promise<boolean>;
}

const ConfirmContext = createContext<ConfirmContextValue | null>(null);

interface PendingState extends ConfirmOptions {
  resolve: Resolver;
}

/**
 * App-level provider that mounts a single `<ConfirmDialog>` lazily and
 * exposes a promise-based `confirm()` API via `useConfirm()`. Replaces
 * blocking `window.confirm()` calls with a real modal that respects
 * focus trap, dark mode, and `prefers-reduced-motion`.
 */
export const ConfirmProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [pending, setPending] = useState<PendingState | null>(null);
  // Track the latest resolver so backdrop/escape dismissal cancels cleanly
  // even if a stale dialog is somehow still in-flight.
  const resolverRef = useRef<Resolver | null>(null);

  const confirm = useCallback((options: ConfirmOptions): Promise<boolean> => {
    return new Promise<boolean>((resolve) => {
      resolverRef.current = resolve;
      setPending({ ...options, resolve });
    });
  }, []);

  const close = useCallback((ok: boolean) => {
    const resolver = resolverRef.current;
    resolverRef.current = null;
    setPending(null);
    resolver?.(ok);
  }, []);

  const value = useMemo<ConfirmContextValue>(() => ({ confirm }), [confirm]);

  return (
    <ConfirmContext.Provider value={value}>
      {children}
      {pending && (
        <ConfirmDialog
          {...pending}
          onConfirm={() => close(true)}
          onCancel={() => close(false)}
        />
      )}
    </ConfirmContext.Provider>
  );
};

export function useConfirm(): ConfirmContextValue['confirm'] {
  const ctx = useContext(ConfirmContext);
  if (!ctx) {
    throw new Error('useConfirm must be used inside <ConfirmProvider>');
  }
  return ctx.confirm;
}

interface ConfirmDialogInternalProps extends ConfirmOptions {
  onConfirm: () => void;
  onCancel: () => void;
}

const ConfirmDialog: React.FC<ConfirmDialogInternalProps> = ({
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'default',
  onConfirm,
  onCancel,
}) => {
  const isDanger = variant === 'danger';
  return (
    <ModalShell
      isOpen
      onClose={onCancel}
      size="md"
      ariaLabelledBy="confirm-dialog-title"
      title={
        <span id="confirm-dialog-title" className="flex items-center gap-2">
          {isDanger && <AlertTriangle className="h-5 w-5 text-red-500" aria-hidden="true" />}
          {title}
        </span>
      }
      footer={
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 focus-ring dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={
              isDanger
                ? 'rounded-md bg-red-600 px-3 py-2 text-sm font-medium text-white hover:bg-red-700 focus-ring'
                : 'rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 focus-ring'
            }
          >
            {confirmLabel}
          </button>
        </div>
      }
    >
      <div className="px-6 py-4 text-sm text-slate-700 dark:text-slate-300">{message}</div>
    </ModalShell>
  );
};
