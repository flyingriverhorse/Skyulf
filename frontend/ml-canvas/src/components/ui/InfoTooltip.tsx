import React from 'react';
import { Info } from 'lucide-react';

interface InfoTooltipProps {
    text: string;
    align?: 'center' | 'left' | 'right';
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({ text, align = 'center' }) => {
    const positionClasses = {
        center: "left-1/2 -translate-x-1/2",
        left: "left-0",
        right: "right-0"
    };

    const arrowClasses = {
        center: "left-1/2 -translate-x-1/2",
        left: "left-2",
        right: "right-2"
    };

    return (
        <div className="group relative ml-2 inline-flex items-center">
            <Info className="w-4 h-4 text-gray-400 cursor-help" />
            <div className={`absolute bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none text-center ${positionClasses[align]}`}>
                {text}
                <div className={`absolute top-full border-4 border-transparent border-t-gray-800 ${arrowClasses[align]}`}></div>
            </div>
        </div>
    );
};
