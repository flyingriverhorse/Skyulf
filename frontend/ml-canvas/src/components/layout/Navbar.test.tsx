import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Navbar } from './Navbar';
import { useViewStore } from '../../core/store/useViewStore';

const originalWidth = window.innerWidth;

beforeEach(() => {
  useViewStore.setState({ activeView: 'canvas', helpGuideTab: null, readOnlyOverride: 'auto' });
});

afterEach(() => {
  Object.defineProperty(window, 'innerWidth', { value: originalWidth, configurable: true });
});

describe('Navbar help button', () => {
  it('keeps tablet edit overrides and selected tabs connected to the real view store', () => {
    // Editing overrides must survive leaving the canvas while the chip stays canvas-only.
    Object.defineProperty(window, 'innerWidth', { value: 900, configurable: true });
    render(<MemoryRouter><Navbar /></MemoryRouter>);
    expect(screen.getByRole('button', { name: 'Read-only' })).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Read-only' }));
    expect(useViewStore.getState().readOnlyOverride).toBe('off');
    expect(screen.getByRole('button', { name: 'Editing' })).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(screen.getByRole('tab', { name: 'Experiments' }));
    expect(useViewStore.getState().activeView).toBe('experiments');
    expect(screen.getByRole('tab', { name: 'Experiments' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.queryByRole('button', { name: 'Editing' })).toBeNull();
    fireEvent.click(screen.getByRole('tab', { name: 'Inference' }));
    expect(useViewStore.getState().activeView).toBe('inference');
    expect(screen.getByRole('tab', { name: 'Inference' })).toHaveAttribute('aria-selected', 'true');
    fireEvent.click(screen.getByRole('tab', { name: 'Canvas' }));
    expect(screen.getByRole('button', { name: 'Editing' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Editing' }));
    expect(useViewStore.getState().readOnlyOverride).toBe('on');
  });

  it('hides the desktop chip after enabling editing from an explicit read-only override', () => {
    // Desktop editing should retain its existing uncluttered navbar after the override changes.
    Object.defineProperty(window, 'innerWidth', { value: 1440, configurable: true });
    useViewStore.setState({ readOnlyOverride: 'on' });
    render(<MemoryRouter><Navbar /></MemoryRouter>);
    fireEvent.click(screen.getByRole('button', { name: 'Read-only' }));
    expect(useViewStore.getState().readOnlyOverride).toBe('off');
    expect(screen.queryByRole('button', { name: 'Read-only' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Editing' })).toBeNull();
  });

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
