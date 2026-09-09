import React from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
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
    return (
        <Tooltip.Provider delayDuration={0}>
            <Tooltip.Root>
                <Tooltip.Trigger asChild>
                    <button type="button" aria-label="Help" className="inline-flex shrink-0 rounded focus-ring">
                        <HelpCircle aria-hidden="true" className="w-3 h-3 text-gray-400 cursor-help" />
                    </button>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                    <Tooltip.Content
                        side={placement === 'bottom-left' ? 'bottom' : 'top'}
                        align={placement === 'bottom-left' ? 'start' : 'center'}
                        sideOffset={6}
                        collisionPadding={8}
                        onEscapeKeyDown={(event) => event.stopPropagation()}
                        className={`z-[200] rounded-md bg-gray-900 p-2.5 text-xs text-white shadow-xl ${placement === 'bottom-left' ? 'w-56' : 'w-48'}`}
                    >
                        {text}
                        <Tooltip.Arrow className="fill-gray-900" />
                    </Tooltip.Content>
                </Tooltip.Portal>
            </Tooltip.Root>
        </Tooltip.Provider>
    );
};
