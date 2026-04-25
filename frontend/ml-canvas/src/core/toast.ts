/**
 * Thin wrapper around `sonner` so call sites import from a stable
 * project path, not a third-party dep. Lets us swap the underlying
 * library (or stub it in tests) without touching ~40 call sites.
 *
 * The `<Toaster />` component is mounted once in `main.tsx`.
 */

import { toast as sonner } from 'sonner';

export const toast = {
  success: (message: string, description?: string): void => {
    sonner.success(message, description ? { description } : undefined);
  },
  error: (message: string, description?: string): void => {
    sonner.error(message, description ? { description } : undefined);
  },
  info: (message: string, description?: string): void => {
    sonner.info(message, description ? { description } : undefined);
  },
  warning: (message: string, description?: string): void => {
    sonner.warning(message, description ? { description } : undefined);
  },
  /** Escape hatch for callers that need full sonner options (custom JSX, etc.). */
  raw: sonner,
};
