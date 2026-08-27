import { describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Navbar } from './Navbar';

describe('Navbar help button', () => {
  it('opens the pipeline guide modal with the concept sections', async () => {
    render(
      <MemoryRouter>
        <Navbar />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByTestId('navbar-help'));

    expect(
      await screen.findByRole('dialog', { name: 'How pipelines work' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: 'After a Split node — order decides' }),
    ).toBeInTheDocument();
  });
});
