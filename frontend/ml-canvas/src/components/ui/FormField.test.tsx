import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { FormField } from './FormField';

describe('FormField', () => {
  it('gives the control an accessible name from the visible label', () => {
    render(
      <FormField label="Encoding Method">
        {(field) => <select {...field}><option>label</option></select>}
      </FormField>,
    );

    expect(screen.getByRole('combobox', { name: 'Encoding Method' })).toBeInTheDocument();
  });

  it('associates the label with the control by id rather than proximity', () => {
    render(
      <FormField label="S3 Path">{(field) => <input {...field} />}</FormField>,
    );

    const input = screen.getByRole('textbox', { name: 'S3 Path' });
    const label = screen.getByText('S3 Path');
    expect(label).toHaveAttribute('for', input.id);
    expect(input.id).not.toBe('');
  });

  it('generates a distinct id for every instance', () => {
    render(
      <>
        <FormField label="First">{(field) => <input {...field} />}</FormField>
        <FormField label="Second">{(field) => <input {...field} />}</FormField>
      </>,
    );

    const first = screen.getByRole('textbox', { name: 'First' });
    const second = screen.getByRole('textbox', { name: 'Second' });
    expect(first.id).not.toBe(second.id);
  });

  it('announces a required field as required before submission', () => {
    render(
      <FormField label="Name" required>{(field) => <input {...field} />}</FormField>,
    );

    const input = screen.getByRole('textbox', { name: /Name/ });
    expect(input).toBeRequired();
  });

  it('does not mark optional fields as required', () => {
    render(
      <FormField label="Access Key">{(field) => <input {...field} />}</FormField>,
    );

    expect(screen.getByRole('textbox', { name: 'Access Key' })).not.toBeRequired();
  });

  it('exposes an invalid state and links the error to the control', () => {
    render(
      <FormField label="S3 Path" error="Must start with s3://">
        {(field) => <input {...field} />}
      </FormField>,
    );

    const input = screen.getByRole('textbox', { name: /S3 Path/ });
    expect(input).toHaveAttribute('aria-invalid', 'true');
    expect(input).toHaveAccessibleDescription('Must start with s3://');
  });

  it('renders the error persistently beside the field, not only for screen readers', () => {
    render(
      <FormField label="S3 Path" error="Must start with s3://">
        {(field) => <input {...field} />}
      </FormField>,
    );

    expect(screen.getByText('Must start with s3://')).toBeVisible();
  });

  it('is not marked invalid when there is no error', () => {
    render(<FormField label="Name">{(field) => <input {...field} />}</FormField>);

    expect(screen.getByRole('textbox', { name: 'Name' })).toHaveAttribute('aria-invalid', 'false');
  });

  it('links a hint to the control so guidance is announced', () => {
    render(
      <FormField label="Target Column" hint="Leave empty for unsupervised runs">
        {(field) => <input {...field} />}
      </FormField>,
    );

    expect(screen.getByRole('textbox', { name: 'Target Column' })).toHaveAccessibleDescription(
      'Leave empty for unsupervised runs',
    );
  });

  it('announces both the hint and the error when a field with guidance fails', () => {
    render(
      <FormField label="Target Column" hint="Pick a column" error="Column not found">
        {(field) => <input {...field} />}
      </FormField>,
    );

    const input = screen.getByRole('textbox', { name: /Target Column/ });
    expect(input).toHaveAccessibleDescription('Pick a column Column not found');
  });

  it('supports labelling a control that must stay visually unlabelled', () => {
    render(
      <FormField label="Filter value" hideLabel>{(field) => <input {...field} />}</FormField>,
    );

    expect(screen.getByRole('textbox', { name: 'Filter value' })).toBeInTheDocument();
  });
});
