import type { EDAProfile } from '../../../core/types/edaProfile';

export type FilterValue = string | number | boolean | Array<string | number>;

export interface FilterItem {
    column: string;
    operator: string;
    value: FilterValue;
}

export interface EDASidebarProps {
    activeTab: string;
    setActiveTab: (tab: string) => void;
    profile: EDAProfile;
    filtersDraft: FilterItem[];
    filtersApplied: FilterItem[];
    filtersDirty: boolean;
    columns: string[];
    excludedCols: string[];
    excludedDirty: boolean;
    analyzing: boolean;
    onAddFilter: (column: string, value: FilterValue, operator: string) => void;
    onRemoveFilter: (index: number) => void;
    onResetFilters: () => void;
    onApplyFilters: () => void;
    onToggleExclude: (column: string, exclude: boolean) => void;
    onApplyExcluded: () => void;
}
