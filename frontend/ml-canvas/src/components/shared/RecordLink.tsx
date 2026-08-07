import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Copy, Check } from 'lucide-react';
import {
  buildRecordHref,
  describeOperationalRef,
  type OperationalContext,
  type OperationalRef,
  type OperationalTimeRange,
} from '../../core/utils/operationalContext';

export interface RecordLinkProps {
  /** The operational record this link addresses. */
  recordRef: OperationalRef;
  /**
   * Visible text. Defaults to the full typed description. When a caller shortens
   * this for layout, the accessible name still carries the untruncated identifier
   * so assistive tech never mistakes a truncated label for the record's identity.
   */
  label?: React.ReactNode;
  /** Route the user is leaving, so the target can offer an accurate return path. */
  origin?: string;
  /** Time window the originating view was scoped to. */
  timeRange?: OperationalTimeRange;
  /** View filters preserved across the handoff. */
  filters?: Record<string, string>;
  /** Renders a button that copies an absolute link to this record. */
  copyable?: boolean;
  className?: string;
}

/**
 * Shared contextual link to any Operations record.
 *
 * Consumers pass a typed reference rather than a hand-built route, so every
 * Operations page produces identical, parseable links without duplicating
 * routing or query-serialization logic.
 */
export const RecordLink: React.FC<RecordLinkProps> = ({
  recordRef,
  label,
  origin,
  timeRange,
  filters,
  copyable = false,
  className = '',
}) => {
  const [copied, setCopied] = useState(false);
  const resetTimerRef = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(resetTimerRef.current), []);

  const context: OperationalContext = {
    ref: recordRef,
    ...(origin !== undefined ? { origin } : {}),
    ...(timeRange !== undefined ? { timeRange } : {}),
    ...(filters !== undefined ? { filters } : {}),
  };

  const href = buildRecordHref(context);
  const description = describeOperationalRef(recordRef);

  const handleCopy = useCallback(async () => {
    const absolute = `${window.location.origin}${href}`;
    try {
      await navigator.clipboard.writeText(absolute);
      setCopied(true);
      window.clearTimeout(resetTimerRef.current);
      resetTimerRef.current = window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard denied or unavailable — the link itself remains usable.
    }
  }, [href]);

  return (
    <span className="inline-flex items-center gap-1">
      <Link
        to={href}
        aria-label={description}
        title={description}
        className={`text-blue-600 hover:underline dark:text-blue-400 ${className}`}
      >
        {label ?? description}
      </Link>
      {copyable && (
        <button
          type="button"
          onClick={handleCopy}
          aria-label={copied ? 'Link copied' : `Copy link to ${description}`}
          className="rounded p-0.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
        </button>
      )}
    </span>
  );
};
