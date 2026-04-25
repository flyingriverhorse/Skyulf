import React, { useState } from 'react';
import { AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { clickableProps } from '../../core/utils/a11y';
import type { EDAAlert } from '../../core/types/edaProfile';

interface AlertsSectionProps {
    alerts: EDAAlert[];
}

export const AlertsSection: React.FC<AlertsSectionProps> = ({ alerts }) => {
    const [isExpanded, setIsExpanded] = useState(true);

    if (!alerts || alerts.length === 0) return null;

    return (
        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
            <div 
                className="flex justify-between items-center cursor-pointer"
                {...clickableProps(() => setIsExpanded(!isExpanded))}
            >
                <h3 className="font-medium text-amber-800 dark:text-amber-200 flex items-center">
                    <AlertCircle className="w-4 h-4 mr-2" />
                    Data Quality Alerts ({alerts.length})
                </h3>
                {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-amber-800 dark:text-amber-200" />
                ) : (
                    <ChevronDown className="w-4 h-4 text-amber-800 dark:text-amber-200" />
                )}
            </div>
            
            {isExpanded && (
                <div className="max-h-40 overflow-y-auto pr-2 mt-2 animate-in slide-in-from-top-2 duration-200">
                    <ul className="space-y-1">
                        {alerts.map((alert, i) => (
                            <li key={i} className="text-sm text-amber-700 dark:text-amber-300 flex items-start">
                                <span className="mr-2">•</span>
                                <span>
                                    {alert.column && <span className="font-semibold">{alert.column}: </span>}
                                    {alert.message}
                                </span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};
