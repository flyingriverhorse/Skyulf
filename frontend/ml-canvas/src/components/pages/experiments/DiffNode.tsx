import React from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeDiffStatus } from '../../../core/utils/graphDiff';
import { nodeDiffRingClass } from '../../../core/utils/graphDiff';
import { NODE_W, NODE_H } from './pipelineDiffLayout';

// Minimal read-only node shown in the side-by-side viewers. The diff
// status is encoded as a ring colour plus a tiny status pill so users
// can read the canvas at a glance. Explicit `Handle` elements are
// required on a custom React Flow node type — without them, edges
// fall back to (0,0) anchor points and visually detach from the node.
export const DiffNode: React.FC<{
  data: { label: string; subLabel?: string; diffStatus: NodeDiffStatus };
}> = ({ data }) => {
  const ring = nodeDiffRingClass(data.diffStatus);
  const pill =
    data.diffStatus === 'added'
      ? { text: 'NEW', cls: 'bg-green-500/15 text-green-600 dark:text-green-400' }
      : data.diffStatus === 'removed'
      ? { text: 'GONE', cls: 'bg-red-500/15 text-red-600 dark:text-red-400' }
      : data.diffStatus === 'modified'
      ? { text: 'EDIT', cls: 'bg-amber-500/15 text-amber-600 dark:text-amber-400' }
      : null;
  return (
    <div
      className={`px-3 py-2 rounded-md border bg-card text-card-foreground shadow-sm flex flex-col justify-center ${ring}`}
      style={{ width: NODE_W, height: NODE_H }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#94a3b8', width: 6, height: 6, border: 'none' }}
      />
      <div className="flex items-center justify-between gap-2 min-w-0">
        <div className="text-xs font-medium truncate">{data.label}</div>
        {pill && (
          <span className={`text-[9px] px-1.5 py-0.5 rounded font-semibold shrink-0 ${pill.cls}`}>
            {pill.text}
          </span>
        )}
      </div>
      {data.subLabel && (
        <div className="text-[10px] text-muted-foreground truncate font-mono">{data.subLabel}</div>
      )}
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#94a3b8', width: 6, height: 6, border: 'none' }}
      />
    </div>
  );
};
