import { VirtualList } from './VirtualList';

export default { title: 'Shared / VirtualList' };

const small = Array.from({ length: 10 }, (_, i) => ({ id: i, label: `Item ${i + 1}` }));
const big = Array.from({ length: 5000 }, (_, i) => ({ id: i, label: `Row ${i + 1}` }));

export const SmallList = () => (
  <div className="p-6">
    <VirtualList
      items={small}
      estimateSize={32}
      getKey={(it) => it.id}
      className="h-64 overflow-auto border border-slate-200 rounded"
      renderItem={(it) => <div className="px-3 py-1 text-sm">{it.label}</div>}
    />
  </div>
);

export const FiveThousandRows = () => (
  <div className="p-6">
    <VirtualList
      items={big}
      estimateSize={28}
      getKey={(it) => it.id}
      className="h-96 overflow-auto border border-slate-200 rounded"
      renderItem={(it) => (
        <div className="px-3 py-1 text-sm border-b border-slate-100">{it.label}</div>
      )}
    />
  </div>
);
