import React from 'react';
import { Info } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../../components/ui/tooltip"; // Assuming we have or will create this shadcn-like component

interface InfoTooltipProps {
    text: string;
    align?: 'center' | 'left' | 'right' | 'end'; // Added 'end' to match usage
    size?: 'sm' | 'md';
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({ text, align = 'center', size = 'md' }) => {
    // If we don't have the shadcn component yet, let's stick to a portal-safe implementation.
    // Ideally, we should use Radix UI TooltipPrimitive to solve clipping.
    // The previous implementation was relative positioning which causes clipping in overflow hidden containers.
    
    // Changing to fixed positioning with standard HTML title is the simplest fallback if libraries are missing, 
    // but the user wants "align and show automatically properly".
    // Let's use standard Radix UI if possible since package.json has @radix-ui/react-tooltip.

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
