import React, { createContext, useContext, useEffect, useId, useRef } from 'react';
import { collectGraphValidationIssues, useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';

interface ValidationTarget {
  requestId: number;
  field?: string | undefined;
  message?: string | undefined;
}
const ValidationContext = createContext<ValidationTarget | null>(null);
const controls = 'input:not([type="hidden"]), select, textarea, button, [tabindex="0"]';

/** Prefer the editable control over auxiliary buttons while a selector loads. */
function fieldControls(field: HTMLElement): HTMLElement[] {
  const inputs = Array.from(field.querySelectorAll<HTMLElement>('input:not([type="hidden"]), select, textarea'));
  return inputs.length ? inputs : Array.from(field.querySelectorAll<HTMLElement>(controls));
}

function canFocus(element: HTMLElement): boolean {
  return !element.matches(':disabled') && !element.closest('[hidden], [inert]')
    && element.checkVisibility?.({ checkVisibilityCSS: true }) !== false;
}

/** Open a local tab or collapsed section once per explicit issue activation. */
export function useValidationReveal(reveal: (field: string) => void): void {
  const target = useContext(ValidationContext);
  const callback = useRef(reveal);
  callback.current = reveal;
  useEffect(() => {
    if (target?.field) callback.current(target.field);
  }, [target?.requestId, target?.field]);
}

/** A stable target shared by a validator and the settings control it describes. */
export function ValidationField({ field, children, className = '' }: {
  field: string; children: React.ReactNode; className?: string;
}) {
  const target = useContext(ValidationContext);
  const message = target?.field === field ? target.message : undefined;
  const errorId = useId();
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const control = root.current && fieldControls(root.current).find(canFocus);
    if (!control || !message) return;
    const invalid = control.getAttribute('aria-invalid');
    const description = control.getAttribute('aria-describedby');
    control.setAttribute('aria-invalid', 'true');
    control.setAttribute('aria-describedby', [description, errorId].filter(Boolean).join(' '));
    return () => {
      if (invalid === null) control.removeAttribute('aria-invalid');
      else control.setAttribute('aria-invalid', invalid);
      if (description === null) control.removeAttribute('aria-describedby');
      else control.setAttribute('aria-describedby', description);
    };
  }, [message, errorId, children]);
  return <div ref={root} data-validation-field={field} tabIndex={-1} role="group" aria-describedby={message ? errorId : undefined}
    className={`min-w-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded-md ${className}`}>
    {children}
    {message && <p id={errorId} className="mt-1 text-xs text-red-700 dark:text-red-300">{message}</p>}
  </div>;
}

/** Navigate once, then leave focus under the user's control as validation updates. */
export function ValidationNavigation({ nodeId, children }: { nodeId: string; children: React.ReactNode }) {
  const request = useViewStore((state) => state.validationFocusRequest);
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const root = useRef<HTMLDivElement>(null);
  const summary = useRef<HTMLDivElement>(null);
  const summaryId = useId();
  const relevant = request?.nodeId === nodeId ? request : null;
  const issue = React.useMemo(() => relevant ? collectGraphValidationIssues(nodes, edges).find((item) =>
    item.nodeId === nodeId && item.category === relevant.category &&
    (relevant.field ? item.field === relevant.field : item.message === relevant.message)
  ) : undefined, [nodes, edges, nodeId, relevant]);
  const target = relevant ? { requestId: relevant.requestId, field: relevant.field, message: issue?.message } : null;

  useEffect(() => {
    if (request && request.nodeId !== nodeId) useViewStore.setState({ validationFocusRequest: null });
  }, [request, nodeId]);

  useEffect(() => {
    const container = root.current;
    return () => {
      // StrictMode's effect replay keeps the DOM mounted; only clear on removal.
      if (!container?.isConnected && useViewStore.getState().validationFocusRequest?.nodeId === nodeId) {
        useViewStore.setState({ validationFocusRequest: null });
      }
    };
  }, [nodeId]);

  useEffect(() => {
    if (!relevant || !root.current) return;
    const container = root.current;
    const focus = (element: HTMLElement) => {
      element.focus({ preventScroll: true });
      element.scrollIntoView?.({ block: 'nearest', inline: 'nearest', behavior: 'instant' });
    };
    // A summary remains useful for connection, cycle, and unmapped errors.
    if (summary.current) focus(summary.current);
    if (!relevant.field) return;
    const tryFocus = () => {
      const field = Array.from(container.querySelectorAll<HTMLElement>('[data-validation-field]'))
        .find((element) => element.dataset.validationField === relevant.field);
      if (!field) return false;
      const candidates = fieldControls(field);
      const control = candidates.find(canFocus);
      if (candidates.length && !control) return false;
      focus(control ?? field);
      return Boolean(control && document.activeElement === control);
    };
    if (tryFocus()) return;
    // Tabs and asynchronous dataset selectors can mount after the request.
    const observer = new MutationObserver(() => { if (tryFocus()) stop(); });
    const stop = () => {
      observer.disconnect();
      document.removeEventListener('pointerdown', stop, true);
      document.removeEventListener('keydown', stop, true);
      document.removeEventListener('focusin', onFocus, true);
    };
    const onFocus = (event: FocusEvent) => {
      if (event.target !== summary.current && !(event.target instanceof HTMLElement && event.target.closest('[data-validation-field]')?.getAttribute('data-validation-field') === relevant.field)) stop();
    };
    observer.observe(container, { childList: true, subtree: true, attributes: true, attributeFilter: ['disabled', 'hidden'] });
    document.addEventListener('pointerdown', stop, true);
    document.addEventListener('keydown', stop, true);
    document.addEventListener('focusin', onFocus, true);
    return stop;
    // Validation changes must clear errors without starting navigation again.
  }, [relevant?.requestId, nodeId]); // eslint-disable-line react-hooks/exhaustive-deps

  return <ValidationContext.Provider value={target}>
    <div ref={root} className="h-full min-h-0 flex flex-col">
      {relevant && (issue || !relevant.field) && <div ref={summary} role="group" aria-label="Validation issue" aria-describedby={summaryId} tabIndex={-1}
        className={`shrink-0 m-3 rounded-md border p-3 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary ${issue ? 'border-red-300 dark:border-red-900' : 'text-muted-foreground'}`}>
        <p id={summaryId}>{issue?.message ?? 'This issue has been resolved.'}</p>
      </div>}
      {children}
    </div>
  </ValidationContext.Provider>;
}
