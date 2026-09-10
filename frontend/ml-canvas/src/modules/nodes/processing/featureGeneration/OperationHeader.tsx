import type React from 'react';
import { ChevronDown, ChevronRight, Trash2 } from 'lucide-react';
import { ARITHMETIC_METHODS, SIMILARITY_METHODS, GROUP_AGG_METHODS } from './options';
import type { MathOperation } from './types';

const METHODS: Record<string, string[]> = {
  arithmetic: ARITHMETIC_METHODS,
  similarity: SIMILARITY_METHODS,
  group_agg: GROUP_AGG_METHODS,
};

interface OperationHeaderProps {
  op: MathOperation;
  idx: number;
  expanded: boolean;
  toggleExpand: (index: number) => void;
  updateOperation: (index: number, updates: Partial<MathOperation>) => void;
  removeOperation: (index: number, event: React.MouseEvent) => void;
}

export function OperationHeader({ op, idx, expanded, toggleExpand, updateOperation, removeOperation }: OperationHeaderProps) {
  const methods = METHODS[op.operation_type] || [];
  return (
    <div
      className="flex items-center justify-between px-3 py-2 bg-muted/30 border-b"
    >
      <div className="flex items-center gap-2">
        <button
          type="button"
          aria-label={`${expanded ? 'Collapse' : 'Expand'} ${op.operation_type === 'datetime_extract' ? 'Date' : op.operation_type.replace('_', ' ')} operation ${idx + 1}`}
          aria-expanded={expanded}
          onClick={() => { toggleExpand(idx); }}
          className="flex items-center gap-2 rounded hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        >
          {expanded ? <ChevronDown size={14} className="text-muted-foreground" /> : <ChevronRight size={14} className="text-muted-foreground" />}
          <span className="text-[10px] font-bold uppercase tracking-wider text-primary bg-primary/10 px-1.5 py-0.5 rounded">
            {op.operation_type === 'datetime_extract' ? 'Date' : op.operation_type.replace('_', ' ')}
          </span>
        </button>
        {['arithmetic', 'similarity', 'group_agg'].includes(op.operation_type) && (
          <select
            aria-label={`Method for operation ${idx + 1}`}
            className="text-sm border-none bg-transparent font-semibold focus:ring-1 focus:ring-primary cursor-pointer hover:text-primary"
            value={op.method}
            onClick={(e) => { e.stopPropagation(); }}
            onChange={(e) => { updateOperation(idx, { method: e.target.value }); }}
          >
            {methods.map(m => (
              <option key={m} value={m}>{m.replace('_', ' ')}</option>
            ))}
          </select>
        )}
      </div>
      <button
        aria-label={`Remove operation ${idx + 1}`}
        onClick={(e) => { removeOperation(idx, e); }}
        className="text-muted-foreground hover:text-destructive transition-colors p-1 rounded-full hover:bg-destructive/10"
      >
        <Trash2 size={14} />
      </button>
    </div>
  );
}
