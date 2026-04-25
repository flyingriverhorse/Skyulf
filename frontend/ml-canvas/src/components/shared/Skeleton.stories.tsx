import { Skeleton, PageSkeleton } from './Skeleton';

export default { title: 'Shared / Skeleton' };

export const Block = () => (
  <div className="p-6">
    <Skeleton className="h-4 w-64" />
  </div>
);

export const Circle = () => (
  <div className="p-6">
    <Skeleton circle className="h-12 w-12" />
  </div>
);

export const PageFallback = () => <PageSkeleton />;
