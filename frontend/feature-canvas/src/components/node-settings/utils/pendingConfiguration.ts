import type { FullExecutionSignal } from '../../../api';

const PENDING_CONFIGURATION_TOKEN = 'pending configuration';
const NOT_CONFIGURED_TOKEN = 'not configured';
const PENDING_REASON_TOKEN = 'pending configuration detected for ';

export type PendingConfigurationDetail = {
  label: string;
  reason: string | null;
};

const sanitizeLabel = (value: string): string => value.replace(/\s+/g, ' ').trim();

const dedupeDetails = (details: PendingConfigurationDetail[]): PendingConfigurationDetail[] => {
  const seen = new Map<string, PendingConfigurationDetail>();
  for (const detail of details) {
    const normalizedLabel = sanitizeLabel(detail.label);
    if (!normalizedLabel) {
      continue;
    }
    const key = normalizedLabel.toLowerCase();
    if (!seen.has(key)) {
      seen.set(key, { label: normalizedLabel, reason: detail.reason?.trim() || null });
    }
  }
  return Array.from(seen.values());
};

const parseStepReason = (step: string): string | null => {
  const [, remainder] = step.split(':', 2);
  if (!remainder) {
    return null;
  }
  const trimmed = remainder.trim();
  if (!trimmed) {
    return null;
  }
  const dashIndex = trimmed.indexOf('–') >= 0 ? trimmed.indexOf('–') : trimmed.indexOf('-');
  if (dashIndex >= 0) {
    return sanitizeLabel(trimmed.slice(dashIndex + 1));
  }
  if (trimmed.toLowerCase().startsWith(PENDING_CONFIGURATION_TOKEN)) {
    return sanitizeLabel(trimmed.slice(PENDING_CONFIGURATION_TOKEN.length));
  }
  return sanitizeLabel(trimmed);
};

const extractFromAppliedSteps = (steps: unknown): PendingConfigurationDetail[] => {
  if (!Array.isArray(steps)) {
    return [];
  }
  const details: PendingConfigurationDetail[] = [];
  for (const step of steps) {
    if (typeof step !== 'string') {
      continue;
    }
    const normalized = step.trim();
    if (
      !normalized ||
      (!normalized.toLowerCase().includes(PENDING_CONFIGURATION_TOKEN) &&
        !normalized.toLowerCase().includes(NOT_CONFIGURED_TOKEN))
    ) {
      continue;
    }
    const label = normalized.split(':', 1)[0]?.trim();
    if (!label) {
      continue;
    }
    details.push({ label, reason: parseStepReason(normalized) });
  }
  return dedupeDetails(details);
};

const extractFromReason = (reason?: string | null): PendingConfigurationDetail[] => {
  if (!reason || typeof reason !== 'string') {
    return [];
  }
  const normalizedReason = reason.toLowerCase();
  const markerIndex = normalizedReason.indexOf(PENDING_REASON_TOKEN);
  if (markerIndex === -1) {
    return [];
  }
  const sliceStart = markerIndex + PENDING_REASON_TOKEN.length;
  const terminalIndex = reason.indexOf('.', sliceStart);
  const summary = reason.substring(sliceStart, terminalIndex === -1 ? reason.length : terminalIndex).trim();
  if (!summary) {
    return [];
  }

  const cleanedSummary = summary.replace(/and \d+ other nodes?$/i, '').trim();
  const pieces = cleanedSummary
    .split(/,| and /i)
    .map((piece) => piece.trim())
    .filter(Boolean);

  return dedupeDetails(pieces.map((label) => ({ label, reason: null })));
};

const extractFromPendingNodesField = (value: unknown): PendingConfigurationDetail[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return dedupeDetails(
    value
      .map((entry) => (typeof entry === 'string' ? entry : ''))
      .filter((entry): entry is string => Boolean(entry))
      .map((entry) => {
        const parts = entry.split(':');
        if (parts.length > 1) {
          const label = parts[0].trim();
          const reason = parts.slice(1).join(':').trim();
          return { label, reason };
        }
        return { label: entry, reason: null };
      })
  );
};

export const extractPendingConfigurationDetails = (
  signal?: FullExecutionSignal | null
): PendingConfigurationDetail[] => {
  if (!signal) {
    return [];
  }

  const nodesFromField = extractFromPendingNodesField((signal as any)?.pending_nodes ?? null);
  if (nodesFromField.length) {
    return nodesFromField;
  }

  const nodesFromSteps = extractFromAppliedSteps(signal.applied_steps);
  if (nodesFromSteps.length) {
    return nodesFromSteps;
  }

  return extractFromReason(signal.reason ?? null);
};
