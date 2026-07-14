import React from 'react';
import type { NodeDiffStatus } from '../../../core/utils/graphDiff';

export const StatusDot: React.FC<{ status: NodeDiffStatus }> = ({ status }) => (
  <span
    className={`inline-block w-2 h-2 rounded-full ${
      status === 'added'
        ? 'bg-green-500'
        : status === 'removed'
        ? 'bg-red-500'
        : status === 'modified'
        ? 'bg-amber-500'
        : 'bg-slate-400'
    }`}
  />
);
