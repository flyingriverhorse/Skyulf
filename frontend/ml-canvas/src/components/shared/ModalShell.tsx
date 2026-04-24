import React, { useEffect } from 'react';
import { X } from 'lucide-react';

export type ModalSize = 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '3xl' | '4xl' | '5xl' | '6xl' | '7xl' | 'full';

const sizeClass: Record<ModalSize, string> = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  '2xl': 'max-w-2xl',
  '3xl': 'max-w-3xl',
  '4xl': 'max-w-4xl',
  '5xl': 'max-w-5xl',
  '6xl': 'max-w-6xl',
  '7xl': 'max-w-7xl',
  full: 'max-w-[95vw]',
};

export interface ModalShellProps {
  isOpen: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  /** Optional content rendered to the right of the title (e.g. action buttons). */
  headerExtra?: React.ReactNode;
  size?: ModalSize;
  /** Tailwind z-index class. Defaults to z-50. Use z-[100] for modals on top of other modals. */
  zIndex?: string;
  /** Click-outside dismisses by default. Set false to require explicit close (e.g. forms). */
  dismissOnBackdrop?: boolean;
  /** Escape dismisses by default. */
  dismissOnEscape?: boolean;
  /** Hide the X close button in the header. */
  hideCloseButton?: boolean;
  /** Override aria-labelledby. Defaults to a generated id when title is a string. */
  ariaLabelledBy?: string;
  className?: string;
  /** Body content. Wrapped in a flex-1 overflow-y-auto container by default. */
  children: React.ReactNode;
  /** Optional footer rendered outside the scrollable body. */
  footer?: React.ReactNode;
}

/**
 * Shared modal wrapper. Centralizes the backdrop, panel, header, close button,
 * Escape handling, and animate-in styles previously duplicated across ~8 modals.
 */
export const ModalShell: React.FC<ModalShellProps> = ({
  isOpen,
  onClose,
  title,
  headerExtra,
  size = '3xl',
  zIndex = 'z-50',
  dismissOnBackdrop = true,
  dismissOnEscape = true,
  hideCloseButton = false,
  ariaLabelledBy,
  className,
  children,
  footer,
}) => {
  useEffect(() => {
    if (!isOpen || !dismissOnEscape) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, dismissOnEscape, onClose]);

  if (!isOpen) return null;

  const titleId = ariaLabelledBy ?? (typeof title === 'string' ? `modal-title-${title.replace(/\s+/g, '-').toLowerCase()}` : undefined);

  return (
    <div
      className={`fixed inset-0 ${zIndex} flex items-center justify-center bg-black/50 backdrop-blur-sm p-4`}
      onClick={(e) => {
        if (dismissOnBackdrop && e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={`bg-white dark:bg-slate-900 rounded-xl shadow-2xl w-full ${sizeClass[size]} max-h-[85vh] flex flex-col border border-slate-200 dark:border-slate-700 animate-in fade-in zoom-in-95 duration-200 ${className ?? ''}`}
      >
        {(title || !hideCloseButton || headerExtra) && (
          <div className="flex items-center justify-between gap-3 p-6 border-b border-slate-200 dark:border-slate-700">
            {title && (
              <h2 id={titleId} className="text-xl font-bold text-slate-900 dark:text-slate-100 flex-1 truncate">
                {title}
              </h2>
            )}
            <div className="flex items-center gap-2">
              {headerExtra}
              {!hideCloseButton && (
                <button
                  onClick={onClose}
                  className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors rounded focus-ring"
                  aria-label="Close"
                  type="button"
                >
                  <X size={24} />
                </button>
              )}
            </div>
          </div>
        )}
        <div className="flex-1 overflow-y-auto">{children}</div>
        {footer && (
          <div className="border-t border-slate-200 dark:border-slate-700 p-4">{footer}</div>
        )}
      </div>
    </div>
  );
};
