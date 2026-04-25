import { EmptyState } from './EmptyState';
import { Database } from 'lucide-react';

export default { title: 'Shared / EmptyState' };

export const Default = () => (
  <EmptyState title="No data yet" description="Upload a dataset to get started." />
);

export const WithAction = () => (
  <EmptyState
    title="No experiments"
    description="Run your first pipeline to see results here."
    action={
      <button className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm">
        Create experiment
      </button>
    }
  />
);

export const CustomIcon = () => (
  <EmptyState
    icon={<Database className="w-12 h-12 text-blue-400" />}
    title="No datasets connected"
    description="Connect a source to begin profiling."
  />
);
