import React, { useRef, ReactNode } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';

interface VirtualListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => ReactNode;
  /** Best-guess row height in px; the virtualizer will measure each row after mount. */
  estimateSize: number;
  /** Below this length we render normally — virtualizing tiny lists is overhead with no payoff. */
  threshold?: number;
  getKey: (item: T, index: number) => string | number;
  /** Class names applied to the scroll viewport (must give it a bounded height). */
  className?: string;
  overscan?: number;
}

/**
 * Drop-in virtualised vertical list. Activates `@tanstack/react-virtual` only
 * when `items.length > threshold` so small lists keep their cheap render path.
 */
export function VirtualList<T>({
  items,
  renderItem,
  estimateSize,
  threshold = 50,
  getKey,
  className,
  overscan = 8,
}: VirtualListProps<T>) {
  const parentRef = useRef<HTMLDivElement>(null);

  // Hooks must run unconditionally; we just ignore the result on the cheap path.
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => estimateSize,
    overscan,
  });

  if (items.length <= threshold) {
    return (
      <div ref={parentRef} className={className}>
        {items.map((item, i) => (
          <React.Fragment key={getKey(item, i)}>{renderItem(item, i)}</React.Fragment>
        ))}
      </div>
    );
  }

  const virtualItems = virtualizer.getVirtualItems();
  return (
    <div ref={parentRef} className={className}>
      <div
        style={{
          height: virtualizer.getTotalSize(),
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualItems.map((vi) => {
          const item = items[vi.index]!;
          return (
            <div
              key={getKey(item, vi.index)}
              data-index={vi.index}
              ref={virtualizer.measureElement}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                transform: `translateY(${vi.start}px)`,
              }}
            >
              {renderItem(item, vi.index)}
            </div>
          );
        })}
      </div>
    </div>
  );
}
