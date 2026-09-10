import { ValidationField } from '../../../../components/shared/ValidationField';
import { OPERATION_TYPES } from './options';
import type { MathOperation } from './types';

export function OperationToolbar({ isWide, addOperation }: { isWide: boolean; addOperation: (type: MathOperation['operation_type']) => void }) {
  return <>
    {isWide ? (
      <ValidationField field="operations" className="w-20 border-r bg-muted/10 flex flex-col items-center py-4 gap-2 overflow-y-auto shrink-0">
        <span className="text-[10px] font-bold text-muted-foreground mb-2 uppercase tracking-wider">Add</span>
        {OPERATION_TYPES.map(t => (
          <button
            key={t.value}
            onClick={() => { addOperation(t.value as MathOperation['operation_type']); }}
            className="flex flex-col items-center justify-center gap-1 p-1.5 rounded-md border bg-card hover:bg-accent hover:border-primary/50 transition-all group w-16 h-14 shadow-sm"
            title={t.label}
          >
            <t.icon size={18} className="text-muted-foreground group-hover:text-primary" />
            <span className="text-[9px] font-medium text-muted-foreground group-hover:text-foreground text-center leading-tight line-clamp-2">
              {t.label.replace(' ', '\n')}
            </span>
          </button>
        ))}
      </ValidationField>
    ) : (
      <ValidationField field="operations" className="flex items-center gap-1.5 border-b bg-muted/10 px-2 py-2 overflow-x-auto shrink-0">
        <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider shrink-0 mr-1">Add:</span>
        {OPERATION_TYPES.map(t => (
          <button
            key={t.value}
            onClick={() => { addOperation(t.value as MathOperation['operation_type']); }}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded border bg-card hover:bg-accent hover:border-primary/50 transition-all group shrink-0 shadow-sm"
            title={t.label}
          >
            <t.icon size={13} className="text-muted-foreground group-hover:text-primary" />
            <span className="text-[10px] font-medium text-muted-foreground group-hover:text-foreground whitespace-nowrap">{t.label}</span>
          </button>
        ))}
      </ValidationField>
    )}
  </>;
}
