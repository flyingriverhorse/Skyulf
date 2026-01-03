import React from 'react';
import { Info } from 'lucide-react';

interface InfoTooltipProps {
    text: string;
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({ text }) => (
    <div className="group relative ml-2 inline-flex items-center">
        <Info className="w-4 h-4 text-gray-400 cursor-help" />
        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none text-center">
            {text}
            <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-800"></div>
        </div>
    </div>
);
