import React from 'react';
import { HelpCircle } from 'lucide-react';

export interface HelpTooltipProps {
    text: string;
    /** Where the tooltip card appears relative to the help icon. Defaults to "top". */
    placement?: 'top' | 'bottom-left';
}

/**
 * Compact help tooltip used in modeling settings panels.
 * Two placements: above the icon (Basic Training style) or below-left (Advanced Tuning style).
 */
export const HelpTooltip: React.FC<HelpTooltipProps> = ({ text, placement = 'top' }) => {
    if (placement === 'bottom-left') {
        return (
            <div className="group relative flex items-center">
                <HelpCircle className="w-3 h-3 text-gray-400 cursor-help" />
                <div className="absolute top-full mt-2 -left-20 hidden group-hover:block w-56 p-2.5 bg-gray-900 text-white text-xs rounded-md shadow-xl z-50">
                    {text}
                    <div className="absolute bottom-full left-20 ml-1.5 border-4 border-transparent border-b-gray-900" />
                </div>
            </div>
        );
    }
    return (
        <div className="group relative flex items-center">
            <HelpCircle className="w-3 h-3 text-gray-400 cursor-help" />
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-48 p-2 bg-gray-900 text-white text-xs rounded shadow-lg z-50">
                {text}
                <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
            </div>
        </div>
    );
};
