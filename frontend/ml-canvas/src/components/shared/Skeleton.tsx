import React from 'react';

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Tailwind size classes, e.g. "h-4 w-32". */
  className?: string;
  /** Render as a circle (avatar / icon placeholder). */
  circle?: boolean;
}

/** Animated placeholder block for async-loading content. */
export const Skeleton: React.FC<SkeletonProps> = ({ className = '', circle = false, ...rest }) => (
  <div
    aria-hidden="true"
    className={`animate-pulse bg-slate-200 dark:bg-slate-700 ${circle ? 'rounded-full' : 'rounded-md'} ${className}`}
    {...rest}
  />
);

/** Page-level skeleton used inside `<Suspense fallback>` for lazy routes. */
export const PageSkeleton: React.FC = () => (
  <div className="flex h-full w-full flex-col gap-4 p-6">
    <Skeleton className="h-8 w-1/3" />
    <Skeleton className="h-4 w-1/2" />
    <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
      <Skeleton className="h-32 w-full" />
      <Skeleton className="h-32 w-full" />
      <Skeleton className="h-32 w-full" />
    </div>
    <Skeleton className="mt-4 h-64 w-full" />
  </div>
);
