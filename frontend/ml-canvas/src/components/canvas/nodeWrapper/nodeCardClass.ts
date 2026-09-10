import type { NodeProps } from '@xyflow/react';
import type { NodePresentation } from './useNodePresentation';

type CardStyle = Pick<NodePresentation, 'leakage' | 'execution' | 'schema' | 'validation' | 'perf'>
  & { selected: NodeProps['selected'] };

function borderClass({ leakage, execution, schema, validation, selected }: CardStyle): string {
  const { leakageSeverity } = leakage;
  const { nodeResult } = execution;
  const { validationMessage } = validation;
  const { hasBrokenRefs } = schema;
  return leakageSeverity === 'error'
    ? 'border-red-500 hover:border-red-500'
    : leakageSeverity === 'warning'
      ? 'border-amber-500 hover:border-amber-500'
      : selected
        ? 'border-primary shadow-lg shadow-primary/30 scale-[1.02]'
        : nodeResult?.status === 'failed'
          ? 'border-red-500 shadow-sm shadow-red-500/20 hover:border-red-500'
          : validationMessage
            ? 'border-red-500/40 hover:border-red-500/60'
            : hasBrokenRefs
              ? 'border-amber-500/40 hover:border-amber-500/60'
              : 'border-border hover:border-primary/50';
}

/** Leakage, selection, execution, validation, and schema warnings retain their precedence. */
export function nodeCardClass(style: CardStyle): string {
  const { selected, leakage, validation, perf } = style;
  const { leakageSeverity } = leakage;
  const { isPulsing } = validation;
  const { perfRingClass } = perf;
  return `
      relative group min-w-[200px] bg-card border-2 rounded-lg shadow-sm transition-all duration-150
      ${borderClass(style)}
      ${leakageSeverity && selected ? 'shadow-lg shadow-primary/30 scale-[1.02]' : ''}
      ${isPulsing && !leakageSeverity ? 'animate-validation-pulse' : ''}
      ${perfRingClass}
    `;
}

/** Keep absent data attributes absent while retaining zero-valued durations. */
export function nodeFeedbackAttributes({ perf, leakage }: Pick<NodePresentation, 'perf' | 'leakage'>) {
  return {
    'data-perf-bucket': perf.perfBucket ?? undefined,
    'data-perf-duration-ms': perf.perfDurationMs ?? undefined,
    'data-leakage-severity': leakage.leakageSeverity ?? undefined,
    title: perf.perfTelemetry?.tooltip,
  };
}
