/** Shared type definitions used by both Basic Training and Advanced Tuning settings. */

export interface HyperparameterDef {
    name: string;
    label: string;
    type: 'number' | 'select' | 'boolean';
    default: unknown;
    description?: string;
    options?: { label: string; value: unknown }[];
    min?: number;
    max?: number;
    step?: number;
}
