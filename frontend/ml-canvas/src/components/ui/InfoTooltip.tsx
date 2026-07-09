import React from 'react';
import { Info } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../../components/ui/tooltip";

interface InfoTooltipProps {
    text: string;
    align?: 'center' | 'left' | 'right' | 'end';
    size?: 'sm' | 'md';
}

/** Small `(i)` icon that reveals a Radix tooltip with `text` on hover/focus. */
export const InfoTooltip: React.FC<InfoTooltipProps> = ({ text, align = 'center', size = 'md' }) => {
    return (
        <TooltipProvider>
            <Tooltip delayDuration={300}>
                <TooltipTrigger asChild>
                    <Info className={`${size === 'sm' ? 'w-3 h-3' : 'w-4 h-4'} text-muted-foreground cursor-help opacity-70 hover:opacity-100 transition-opacity`} />
                </TooltipTrigger>
                <TooltipContent side="top" align={align === 'end' ? 'end' : 'center'} className="max-w-xs text-xs">
                    <p>{text}</p>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );
};
