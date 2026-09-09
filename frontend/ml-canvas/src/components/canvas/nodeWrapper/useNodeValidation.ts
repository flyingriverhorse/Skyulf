import { useEffect, useMemo, useRef, useState } from 'react';
import type { NodeDefinition } from '../../../core/types/nodes';

/** Validate configuration safely and pulse only when it becomes invalid. */
export function useNodeValidation(definition: NodeDefinition<unknown> | undefined, data: Record<string, unknown>) {
  let validationMessage: string | null = null;
  const validationResult = useMemo(() => {
    if (!definition) return null;
    try {
      return definition.validate(data);
    } catch {
      // Validator threw — treat as a soft warning, don't block rendering.
      return null;
    }
  }, [definition, data]);
  if (validationResult && !validationResult.isValid) {
    validationMessage = validationResult.message ?? 'Configuration incomplete.';
  }

  // One-shot pulse animation: when a node transitions from valid →
  // invalid, play a 5 s red-ring pulse to draw attention. `isPulsing`
  // gates the CSS class; we only set it on the false→true edge of
  // `isInvalid`, never every render, so an already-invalid node
  // doesn't re-pulse forever. The class auto-clears after 5 s.
  const wasInvalidRef = useRef<boolean>(false);
  const [isPulsing, setIsPulsing] = useState<boolean>(false);
  const isInvalid = validationMessage !== null;
  useEffect(() => {
    if (isInvalid && !wasInvalidRef.current) {
      setIsPulsing(true);
      // Matches the CSS animation: 3 cycles × 1.6 s = 4.8 s.
      const t = window.setTimeout(() => setIsPulsing(false), 4800);
      wasInvalidRef.current = true;
      return () => window.clearTimeout(t);
    }
    if (!isInvalid && wasInvalidRef.current) {
      // Cleared by the user fixing config — drop the pulse early.
      wasInvalidRef.current = false;
      setIsPulsing(false);
    }
    return undefined;
  }, [isInvalid]);
  return { validationMessage, isPulsing };
}
