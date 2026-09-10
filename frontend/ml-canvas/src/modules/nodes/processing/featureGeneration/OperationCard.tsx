import type React from 'react';
import { OperationEditor } from './OperationEditors';
import { OperationHeader } from './OperationHeader';
import type { OperationEditorProps } from './types';

interface OperationCardProps extends OperationEditorProps {
  revealedOperations: number[];
  toggleExpand: (index: number) => void;
  removeOperation: (index: number, event: React.MouseEvent) => void;
}

export function OperationCard(props: OperationCardProps) {
  const { op, idx, updateOperation, revealedOperations } = props;
  const expanded = !!(op.isExpanded || revealedOperations.includes(idx));
  return (
    <div className="border rounded-lg bg-card shadow-sm overflow-hidden">
      <OperationHeader {...props} expanded={expanded} />
      {expanded && <div className="p-3 space-y-4">
        <OperationEditor {...props} />
        {/* Output Name */}
        <div className="pt-2">
          <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1 block">Output Column Name</span>
          <input
            aria-label={`Output Column Name for operation ${idx + 1}`}
            type="text"
            className="w-full text-xs border rounded px-2 py-1.5 bg-background focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
            placeholder={`${op.operation_type}_${idx}`}
            value={op.output_column || ''}
            onChange={(e) => { updateOperation(idx, { output_column: e.target.value }); }}
          />
        </div>
      </div>}
    </div>
  );
}
