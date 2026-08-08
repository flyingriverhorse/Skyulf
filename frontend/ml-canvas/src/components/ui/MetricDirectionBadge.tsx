import React from 'react';
import { ArrowDown, ArrowUp, HelpCircle } from 'lucide-react';

import type { MetricDirection } from '../../core/utils/metricMeta';

interface Props {
  direction: MetricDirection;
}

const COPY: Record<MetricDirection, { label: string; className: string }> = {
  higher: { label: 'Higher is better', className: 'text-emerald-600 dark:text-emerald-400' },
  lower: { label: 'Lower is better', className: 'text-sky-600 dark:text-sky-400' },
  unknown: { label: 'Direction unknown — not ranked', className: 'text-gray-500 dark:text-gray-400' },
};

/**
 * Renders whether a larger or smaller value of a metric is better, so the
 * reading of a comparison never depends on a tooltip the user must discover.
 */
export const MetricDirectionBadge: React.FC<Props> = ({ direction }) => {
  const { label, className } = COPY[direction];
  const Icon = direction === 'higher' ? ArrowUp : direction === 'lower' ? ArrowDown : HelpCircle;

  return (
    <span className={`inline-flex items-center gap-0.5 text-[10px] font-medium ${className}`} title={label}>
      <Icon className="w-3 h-3" aria-hidden="true" />
      <span className="sr-only">{label}</span>
      <span aria-hidden="true">{direction === 'unknown' ? '?' : direction === 'higher' ? 'higher' : 'lower'}</span>
    </span>
  );
};
