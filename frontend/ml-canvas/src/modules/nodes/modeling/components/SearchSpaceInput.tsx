import React, { useEffect, useState } from 'react';
import { AlertCircle } from 'lucide-react';
import { HelpTooltip } from './HelpTooltip';
import type { HyperparameterDef } from './types';

export interface SearchSpaceInputProps {
    def: HyperparameterDef;
    value: unknown[];
    onChange: (values: unknown[]) => void;
}

/**
 * Multi-value (search-space) input used by AdvancedTuningSettings.
 * Comma-separated entries get parsed/validated against the param's type.
 * For `select` defs, option chips toggle inclusion in the value list.
 */
export const SearchSpaceInput: React.FC<SearchSpaceInputProps> = ({ def, value, onChange }) => {
    const [localValue, setLocalValue] = useState('');
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        setLocalValue(Array.isArray(value) ? value.map(v => v === null ? 'None' : v).join(', ') : '');
    }, [value]);

    const validateAndParse = (input: string): unknown[] => {
        if (!input.trim()) return [];

        const parts = input.split(',').map(s => s.trim()).filter(s => s !== '');
        const parsed: unknown[] = [];

        for (const part of parts) {
            if (part.toLowerCase() === 'none') {
                parsed.push(null);
                continue;
            }

            if (def.type === 'number') {
                const num = Number(part);
                if (isNaN(num)) {
                    throw new Error(`"${part}" is not a valid number`);
                }
                parsed.push(num);
            } else if (def.type === 'boolean') {
                const lower = part.toLowerCase();
                if (lower === 'true') parsed.push(true);
                else if (lower === 'false') parsed.push(false);
                else throw new Error(`"${part}" must be true or false`);
            } else {
                parsed.push(part);
            }
        }
        return parsed;
    };

    const handleBlur = () => {
        try {
            const parsed = validateAndParse(localValue);
            setError(null);
            // Avoid identity-only re-renders that could loop upstream effects
            if (JSON.stringify(parsed) !== JSON.stringify(value)) {
                onChange(parsed);
            }
        } catch (err: unknown) {
            setError((err as Error).message);
        }
    };

    return (
        <div className="space-y-1">
            <div className="flex justify-between items-center">
                <label className="text-xs font-medium text-gray-700 dark:text-gray-300 flex items-center gap-1">
                    {def.label}
                    {def.description && <HelpTooltip text={def.description} placement="bottom-left" />}
                </label>
                <span className="text-[10px] text-gray-400 uppercase">{def.type}</span>
            </div>

            <div className="relative">
                <input
                    type="text"
                    className={`w-full border rounded px-2 py-1.5 text-sm font-mono bg-white dark:bg-gray-900 dark:text-gray-100 outline-none transition-all ${
                        error
                            ? 'border-red-500 focus:ring-2 focus:ring-red-500/20'
                            : 'border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500'
                    }`}
                    placeholder={def.type === 'select' ? "e.g. lbfgs, liblinear" : "e.g. 10, 50, 100"}
                    value={localValue}
                    onChange={(e) => {
                        setLocalValue(e.target.value);
                        setError(null);
                    }}
                    onBlur={handleBlur}
                />
                {def.type === 'select' && def.options && (
                    <div className="mt-1 flex flex-wrap gap-1">
                        {def.options.map(opt => (
                            <button
                                key={String(opt.value)}
                                onClick={() => {
                                    const current = validateAndParse(localValue);
                                    const exists = current.includes(opt.value);
                                    const newValue = exists
                                        ? current.filter(v => v !== opt.value)
                                        : [...current, opt.value];
                                    onChange(newValue);
                                }}
                                className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
                                    value.includes(opt.value)
                                        ? 'bg-purple-100 border-purple-200 text-purple-700 dark:bg-purple-900/30 dark:border-purple-800 dark:text-purple-300'
                                        : 'bg-gray-50 border-gray-200 text-gray-600 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                                }`}
                            >
                                {opt.label}
                            </button>
                        ))}
                    </div>
                )}
            </div>
            {error && (
                <p className="text-[10px] text-red-500 flex items-center gap-1">
                    <AlertCircle className="w-3 h-3" />
                    {error}
                </p>
            )}
        </div>
    );
};
