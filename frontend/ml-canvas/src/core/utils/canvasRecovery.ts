/**
 * CAN-003: shared vocabulary for explaining Canvas recovery sources.
 *
 * The Canvas graph can be restored from three places — the local autosave
 * (`canvasPersistence.ts`), the per-browser Recent list
 * (`recentPipelines.ts`), and server-side pipeline versions
 * (`api/pipelineVersions.ts`) — each with its own storage shape. This module
 * gives them one shared `RecoverySourceKind` label and turns an autosave
 * diagnostic into a plain-language reason instead of a silent no-op, so a
 * user can tell "nothing was saved" apart from "the snapshot can't be used".
 */
import type { CanvasSnapshotDiagnostic } from './canvasPersistence';

/** Where a candidate graph to restore came from. */
export type RecoverySourceKind = 'autosave' | 'local-recent' | 'server-version';

/** Human-facing label for a recovery source kind, used in confirmations and
 *  source-labelled lists so a user always knows whether they're about to
 *  load a browser-local snapshot or a server-backed one. */
export const RECOVERY_KIND_LABEL: Record<RecoverySourceKind, string> = {
  autosave: 'Autosave',
  'local-recent': 'Local recent',
  'server-version': 'Server version',
};

/** Reasons an autosave snapshot exists but can't be restored. Mirrors the
 *  non-`available`/`empty` statuses of `CanvasSnapshotDiagnostic`. */
export type AutosaveUnavailableStatus = 'corrupt' | 'version-mismatch' | 'storage-error';

export interface AutosaveUnavailable {
  status: AutosaveUnavailableStatus;
  /** Plain-language explanation safe to show directly in the UI — never
   *  exposes the raw storage key, stack trace, or parsed payload shape. */
  message: string;
}

const AUTOSAVE_UNAVAILABLE_MESSAGES: Record<AutosaveUnavailableStatus, string> = {
  corrupt:
    'A previous autosave could not be read because the saved data was corrupted, so nothing was restored.',
  'version-mismatch':
    'A previous autosave was made by an incompatible version of Canvas and was skipped.',
  'storage-error':
    "Autosave could not be read from this browser's local storage — it may be full or disabled.",
};

/** Turn a `canvasPersistence` diagnostic into a user-facing explanation.
 *  Returns `null` when there's nothing to explain (a snapshot is available,
 *  or none was ever saved). */
export function describeAutosaveUnavailable(
  diagnostic: CanvasSnapshotDiagnostic,
): AutosaveUnavailable | null {
  if (diagnostic.status === 'available' || diagnostic.status === 'empty') return null;
  return { status: diagnostic.status, message: AUTOSAVE_UNAVAILABLE_MESSAGES[diagnostic.status] };
}
