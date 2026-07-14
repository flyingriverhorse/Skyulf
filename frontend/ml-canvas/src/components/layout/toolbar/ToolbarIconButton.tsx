import React from 'react';

interface ToolbarIconButtonProps {
  icon: React.ReactNode;
  onClick: () => void;
  title: string;
  ariaLabel: string;
  disabled?: boolean;
  /** Danger variant swaps the hover color to red (e.g. destructive actions like "Clear canvas"). */
  variant?: 'default' | 'danger';
  /** Most icon buttons want the shared focus ring; omit only to match a pre-existing exception. */
  focusRing?: boolean;
  ariaExpanded?: boolean;
  testId?: string;
}

/**
 * Shared square (40x40) icon-only button used by the Toolbar's left cluster
 * (legend, keyboard shortcuts, command palette, undo/redo, clear canvas).
 * Extracted because these ~6 buttons previously repeated the same long
 * className string with only the icon/handler/aria attributes differing.
 */
export const ToolbarIconButton: React.FC<ToolbarIconButtonProps> = ({
  icon,
  onClick,
  title,
  ariaLabel,
  disabled,
  variant = 'default',
  focusRing = true,
  ariaExpanded,
  testId,
}) => {
  const hoverClasses =
    variant === 'danger'
      ? 'hover:bg-red-50 hover:text-red-600 hover:border-red-300 dark:hover:bg-red-950/30'
      : 'hover:bg-accent';

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      aria-expanded={ariaExpanded}
      data-testid={testId}
      className={`flex items-center justify-center w-10 h-10 bg-background border rounded-md shadow-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${hoverClasses} ${focusRing ? 'focus-ring' : ''}`}
    >
      {icon}
    </button>
  );
};
