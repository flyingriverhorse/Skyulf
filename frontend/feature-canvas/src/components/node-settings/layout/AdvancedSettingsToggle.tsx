import React from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

type AdvancedSettingsToggleProps = {
  isOpen: boolean;
  onToggle: () => void;
  label?: string;
  description?: string;
};

export const AdvancedSettingsToggle: React.FC<AdvancedSettingsToggleProps> = ({
  isOpen,
  onToggle,
  label = 'Advanced settings',
  description,
}) => {
  return (
    <div className="canvas-modal__advanced-toggle">
      <button
        type="button"
        className="canvas-modal__advanced-button"
        onClick={onToggle}
        aria-expanded={isOpen}
      >
        {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <span>{label}</span>
      </button>
      {description && !isOpen && (
        <span className="canvas-modal__advanced-description">{description}</span>
      )}
    </div>
  );
};
