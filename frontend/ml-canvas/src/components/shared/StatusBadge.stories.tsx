import { StatusBadge } from './StatusBadge';

export default { title: 'Shared / StatusBadge' };

const ALL: Array<'completed' | 'failed' | 'running' | 'pending' | 'cancelled' | 'idle'> = [
  'completed',
  'failed',
  'running',
  'pending',
  'cancelled',
  'idle',
];

export const AllStatuses = () => (
  <div className="flex flex-wrap gap-3 p-6">
    {ALL.map((s) => (
      <StatusBadge key={s} status={s} />
    ))}
  </div>
);

export const IconOnly = () => (
  <div className="flex flex-wrap gap-3 p-6">
    {ALL.map((s) => (
      <StatusBadge key={s} status={s} iconOnly />
    ))}
  </div>
);

export const TextOnly = () => (
  <div className="flex flex-wrap gap-3 p-6">
    {ALL.map((s) => (
      <StatusBadge key={s} status={s} textOnly />
    ))}
  </div>
);

export const UnknownFallback = () => (
  <div className="p-6">
    <StatusBadge status="weird-state" />
  </div>
);
