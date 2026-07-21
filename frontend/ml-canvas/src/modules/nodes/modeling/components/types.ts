/** Shared type definitions used by both Basic Training and Advanced Tuning settings. */

export interface HyperparameterDef {
    name: string;
    label: string;
    type: 'number' | 'select' | 'boolean' | 'multiselect';
    default: unknown;
    description?: string;
    options?: { label: string; value: unknown }[];
    min?: number;
    max?: number;
    step?: number;
    /** Only relevant when another param equals a given value, e.g. `l1_ratio` needs `penalty === 'elasticnet'`. */
    depends_on?: { param: string; value: unknown };
    /** These option values can't be combined with any other option in the same search space (e.g. `elasticnet` vs `l1`/`l2`). */
    exclusive_options?: unknown[];
}
