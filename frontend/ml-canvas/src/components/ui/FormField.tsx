import React, { useId } from 'react';

/** Props a `FormField` hands to its control so the association is never left to proximity. */
export interface FieldControlProps {
  id: string;
  required: boolean;
  'aria-invalid': boolean;
  'aria-describedby'?: string;
}

interface Props {
  label: string;
  /** Renders the label for assistive technology only, for controls whose purpose is visually obvious from context. */
  hideLabel?: boolean;
  required?: boolean;
  /** Persistent guidance rendered beside the field and announced as a description. */
  hint?: string;
  /** Validation failure. Presence switches the control into an invalid state. */
  error?: string;
  className?: string;
  children: (field: FieldControlProps) => React.ReactNode;
}

/**
 * Wraps a single form control with a programmatically associated label, required
 * state, hint, and error, so keyboard and assistive-technology users get the same
 * information sighted users read from layout alone.
 */
export const FormField: React.FC<Props> = ({
  label,
  hideLabel = false,
  required = false,
  hint,
  error,
  className,
  children,
}) => {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;

  const describedBy = [hint ? hintId : null, error ? errorId : null].filter(Boolean).join(' ');

  const field: FieldControlProps = {
    id,
    required,
    'aria-invalid': Boolean(error),
    ...(describedBy ? { 'aria-describedby': describedBy } : {}),
  };

  return (
    <div className={className}>
      <label
        htmlFor={id}
        className={
          hideLabel
            ? 'sr-only'
            : 'block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1'
        }
      >
        {label}
        {required && !hideLabel && (
          <span className="text-red-500 ml-0.5" aria-hidden="true">*</span>
        )}
      </label>

      {children(field)}

      {hint && (
        <p id={hintId} className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          {hint}
        </p>
      )}
      {error && (
        <p id={errorId} className="mt-1 text-xs text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
    </div>
  );
};
