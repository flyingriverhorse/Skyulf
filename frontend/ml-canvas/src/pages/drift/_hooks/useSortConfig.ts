import { useCallback, useState } from 'react';

export interface SortConfig {
    key: string;
    dir: 'asc' | 'desc';
}

/**
 * Tri-state sort: first click → ascending, second → descending, third → cleared.
 * Returned from `useSortConfig` so each consumer (e.g. the drift table) can
 * drive its own column header clicks.
 */
export function useSortConfig() {
    const [sortConfig, setSortConfig] = useState<SortConfig | null>(null);

    const handleSort = useCallback((key: string) => {
        setSortConfig(prev => {
            if (prev?.key === key) {
                if (prev.dir === 'asc') return { key, dir: 'desc' };
                return null;
            }
            return { key, dir: 'asc' };
        });
    }, []);

    const clearSort = useCallback(() => setSortConfig(null), []);

    return { sortConfig, handleSort, clearSort };
}
