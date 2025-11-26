import React from 'react';

type FormFieldProps = {
  label: string;
  htmlFor?: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
};

export const FormField: React.FC<FormFieldProps> = ({
  label,
  htmlFor,
  description,
  children,
  className = '',
}) => {
  return (
    <div className={`canvas-modal__parameter-field ${className}`}>
      <label htmlFor={htmlFor} className="canvas-modal__parameter-label">
        {label}
      </label>
      <div className="canvas-modal__parameter-control">{children}</div>
      {description && <p className="canvas-modal__parameter-description">{description}</p>}
    </div>
  );
};

type TextInputProps = React.InputHTMLAttributes<HTMLInputElement>;

export const TextInput: React.FC<TextInputProps> = ({ className = '', ...props }) => {
  return <input type="text" className={`canvas-modal__input ${className}`} {...props} />;
};

type NumberInputProps = React.InputHTMLAttributes<HTMLInputElement>;

export const NumberInput: React.FC<NumberInputProps> = ({ className = '', ...props }) => {
  return <input type="number" className={`canvas-modal__input ${className}`} {...props} />;
};

type SelectInputProps = React.SelectHTMLAttributes<HTMLSelectElement> & {
  options: { value: string | number; label: string }[];
};

export const SelectInput: React.FC<SelectInputProps> = ({
  className = '',
  options,
  children,
  ...props
}) => {
  return (
    <select className={`canvas-modal__select ${className}`} {...props}>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
      {children}
    </select>
  );
};

type CheckboxInputProps = React.InputHTMLAttributes<HTMLInputElement> & {
  label: string;
  fieldLabel?: string;
  description?: string;
};

export const CheckboxInput: React.FC<CheckboxInputProps> = ({
  label,
  fieldLabel,
  description,
  className = '',
  ...props
}) => {
  return (
    <div className={`canvas-modal__parameter-field ${className}`}>
      {fieldLabel && (
        <div className="canvas-modal__parameter-label">
          <label htmlFor={props.id}>{fieldLabel}</label>
        </div>
      )}
      <label className="canvas-modal__boolean-control">
        <input type="checkbox" {...props} />
        {label}
      </label>
      {description && <p className="canvas-modal__parameter-description">{description}</p>}
    </div>
  );
};

type RangeInputProps = React.InputHTMLAttributes<HTMLInputElement>;

export const RangeInput: React.FC<RangeInputProps> = ({ className = '', ...props }) => {
  return <input type="range" className={`canvas-imputer__filter-range ${className}`} {...props} />;
};

type TextAreaInputProps = React.TextareaHTMLAttributes<HTMLTextAreaElement>;

export const TextAreaInput: React.FC<TextAreaInputProps> = ({ className = '', ...props }) => {
  return <textarea className={`canvas-modal__input ${className}`} {...props} />;
};
