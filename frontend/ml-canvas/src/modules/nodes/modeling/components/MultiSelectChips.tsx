import { Check } from 'lucide-react';

interface Option {
    label: string;
    value: unknown;
}

interface MultiSelectChipsProps {
    options: Option[];
    /** Currently selected values (order preserved as the user toggles). */
    selected: unknown[];
    onChange: (next: unknown[]) => void;
}

/**
 * Chip-style multi-select used for list-valued hyperparameters (e.g. an
 * ensemble's base models). Toggling a chip adds/removes its value; selection
 * order is preserved so the rendered estimator order matches the user's intent.
 */
export function MultiSelectChips({ options, selected, onChange }: MultiSelectChipsProps) {
    const toggle = (value: unknown) => {
        if (selected.includes(value)) {
            onChange(selected.filter((v) => v !== value));
        } else {
            onChange([...selected, value]);
        }
    };

    return (
        <div className="flex flex-wrap gap-1.5">
            {options.map((opt) => {
                const active = selected.includes(opt.value);
                return (
                    <button
                        key={String(opt.value)}
                        type="button"
                        onClick={() => { toggle(opt.value); }}
                        className={`flex items-center gap-1 px-2 py-1 rounded-lg text-xs border transition-colors ${
                            active
                                ? 'bg-blue-500 text-white border-blue-500'
                                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
                        }`}
                    >
                        {active && <Check size={11} className="shrink-0" />}
                        {opt.label}
                    </button>
                );
            })}
        </div>
    );
}
