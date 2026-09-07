import { beforeEach, describe, expect, it } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { NotificationCenter } from '../components/layout/NotificationCenter';
import { useNotificationsStore } from './store/useNotificationsStore';
import { toast } from './toast';

beforeEach(() => useNotificationsStore.getState().clear());

describe('notification center announcements', () => {
  it('announces a repeated failure after reading it without accumulating duplicate entries', () => {
    // A failed retry needs fresh feedback even when the backend returns the same message.
    render(<MemoryRouter><NotificationCenter /></MemoryRouter>);
    act(() => toast.error('Export failed'));
    fireEvent.click(screen.getByRole('button', { name: /Notifications/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(useNotificationsStore.getState().items[0]?.read).toBe(true);
    act(() => toast.error('Export failed'));
    expect(useNotificationsStore.getState().items).toHaveLength(1);
    expect(useNotificationsStore.getState().items[0]?.read).toBe(false);
    expect(screen.getByTestId('notification-announcement')).toHaveTextContent('Export failed');
  });

  it.each(['success', 'error', 'info', 'warning'] as const)('stores and announces %s messages without opening a popup', (level) => {
    // Removing toast popups must preserve messages, descriptions, and screen-reader feedback.
    render(<MemoryRouter><NotificationCenter /></MemoryRouter>);
    act(() => toast[level]('Dataset saved', 'Ready to use.'));
    expect(useNotificationsStore.getState().items).toHaveLength(1);
    expect(useNotificationsStore.getState().items[0]).toMatchObject({ level, message: 'Dataset saved\nReady to use.' });
    expect(screen.getByTestId('notification-announcement')).toHaveTextContent('Dataset saved Ready to use.');
    expect(screen.queryByRole('button', { name: 'Clear all' })).toBeNull();
    expect(document.querySelector('[data-sonner-toaster]')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /Notifications/ }));
    expect(screen.getByText('Dataset saved Ready to use.')).toBeInTheDocument();
    expect(screen.queryByText('Click to see full details')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));
    expect(useNotificationsStore.getState().items).toHaveLength(0);
  });
});
