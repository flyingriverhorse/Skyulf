import { beforeEach, expect, it } from 'vitest';
import { useNotificationsStore } from './useNotificationsStore';

beforeEach(() => useNotificationsStore.getState().clear());

it('updates an execution entry without duplicates and marks changed messages unread', () => {
  // Repeated blocked runs must replace their notice without repeatedly flagging identical text.
  const store = useNotificationsStore.getState();
  store.upsertExecution('canvas-preview', 'Preview blocked', { type: 'preview' }, 'warning');
  store.markAllRead();
  store.upsertExecution('canvas-preview', 'Preview blocked', { type: 'preview' }, 'warning');
  expect(useNotificationsStore.getState().items).toHaveLength(1);
  expect(useNotificationsStore.getState().items[0]?.read).toBe(true);
  store.upsertExecution('canvas-preview', 'Preview failed', { type: 'preview' }, 'error');
  expect(useNotificationsStore.getState().items[0]).toMatchObject({ message: 'Preview failed', level: 'error', read: false });
});

it('caps execution history and supports individual and complete clearing', () => {
  // Execution receipts share the existing bounded, dismissible notification history.
  const store = useNotificationsStore.getState();
  for (let i = 0; i < 105; i++) {
    store.upsertExecution(`run-${i}`, 'Submitted', { type: 'jobs', run: { label: 'Experiments', jobIds: [`job-${i}`] } });
  }
  expect(useNotificationsStore.getState().items).toHaveLength(100);
  expect(useNotificationsStore.getState().items[0]).toMatchObject({ id: 'run-104', level: 'info' });
  store.dismiss('run-104');
  expect(useNotificationsStore.getState().items).toHaveLength(99);
  store.clear();
  expect(useNotificationsStore.getState().items).toEqual([]);
});

it('persists execution actions as data alongside existing warning entries', () => {
  // Reloaded bell entries need their destination without persisting callbacks.
  const store = useNotificationsStore.getState();
  store.addMany([{ node_id: 'a', node_type: 'Scaler', message: 'Warning', level: 'warning', logger: 'pipeline' }]);
  store.upsertExecution('run', 'Submitted', { type: 'jobs', run: { label: 'Experiments', jobIds: ['job'] } });
  const persisted = JSON.parse(localStorage.getItem('skyulf-notifications') || '{}');
  expect(persisted.state.items[0].action).toEqual({ type: 'jobs', run: { label: 'Experiments', jobIds: ['job'] } });
  expect(persisted.state.items[1].message).toBe('Warning');
});
