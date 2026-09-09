import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { selectUnreadCount, useNotificationsStore } from './useNotificationsStore';

beforeEach(() => useNotificationsStore.getState().clear());
afterEach(() => vi.restoreAllMocks());

it('keeps distinct app messages and warnings when clock and nonsecure randomness repeat', () => {
  // ID collisions must not overwrite messages or make dismissing one remove another.
  vi.spyOn(Date, 'now').mockReturnValue(1000);
  vi.spyOn(Math, 'random').mockReturnValue(0);
  const store = useNotificationsStore.getState();
  store.addAppMessage('info', 'Dataset saved');
  store.addAppMessage('info', 'Model saved');
  store.addMany(['Missing values', 'Constant column'].map(message => ({
    node_id: null, node_type: null, level: 'warning', message, logger: 'pipeline',
  })));

  const items = useNotificationsStore.getState().items;
  expect(items).toHaveLength(4);
  expect(new Set(items.map(item => item.id)).size).toBe(4);
  store.dismiss(items[0]!.id);
  expect(useNotificationsStore.getState().items.map(item => item.message)).toEqual([
    'Constant column', 'Model saved', 'Dataset saved',
  ]);
});

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

it('refreshes a receipt destination without moving unchanged read messages above newer items', () => {
  // Updating job IDs alone must preserve reading state and chronology while fixing navigation.
  const now = vi.spyOn(Date, 'now').mockReturnValue(1000);
  const store = useNotificationsStore.getState();
  store.upsertExecution('older', 'Submitted', { type: 'jobs', run: { label: 'Run', jobIds: ['old'] } });
  now.mockReturnValue(2000);
  store.upsertExecution('newer', 'Preview ready', { type: 'preview' });
  store.markAllRead();
  now.mockReturnValue(3000);
  store.upsertExecution('older', 'Submitted', {
    type: 'jobs', run: { label: 'Run', jobIds: ['old', 'replacement'] },
  });

  expect(useNotificationsStore.getState().items.map(item => item.id)).toEqual(['newer', 'older']);
  expect(useNotificationsStore.getState().items[1]).toMatchObject({
    ts: 1000, read: true, action: { type: 'jobs', run: { jobIds: ['old', 'replacement'] } },
  });
  expect(selectUnreadCount(useNotificationsStore.getState())).toBe(0);
});

it('makes severity changes unread even when the message text stays the same', () => {
  // A warning becoming an error deserves a new alert even if the server reuses its text.
  const store = useNotificationsStore.getState();
  store.upsertExecution('execution', 'Run blocked', { type: 'preview' }, 'warning');
  store.markAllRead();
  store.upsertExecution('execution', 'Run blocked', { type: 'preview' }, 'error');

  expect(useNotificationsStore.getState().items).toHaveLength(1);
  expect(useNotificationsStore.getState().items[0]).toMatchObject({ level: 'error', read: false });
  expect(selectUnreadCount(useNotificationsStore.getState())).toBe(1);
});

it('deduplicates warnings without node metadata within a batch and across refreshes', () => {
  // Missing node IDs must not flood the bell, but the same warning on another node is distinct.
  const store = useNotificationsStore.getState();
  const warning = { node_id: null, node_type: null, level: 'warning', message: 'No split', logger: 'pipeline' };
  store.addMany([warning, warning, { ...warning, node_id: 'splitter' }]);
  store.markAllRead();
  const previous = useNotificationsStore.getState().items;
  store.addMany([]);
  store.addMany([warning]);

  expect(useNotificationsStore.getState().items).toBe(previous);
  expect(useNotificationsStore.getState().items).toHaveLength(2);
  expect(selectUnreadCount(useNotificationsStore.getState())).toBe(0);
});

it('refreshes repeated app messages without replacing warnings from other sources', () => {
  // Identical text from a pipeline and the app needs separate context, while repeated app alerts reuse one row.
  const store = useNotificationsStore.getState();
  store.addMany([{ node_id: null, node_type: null, level: 'warning', message: 'Unavailable', logger: 'pipeline' }]);
  store.addAppMessage('warning', 'Unavailable');
  const appId = useNotificationsStore.getState().items[0]?.id;
  store.markAllRead();
  store.addAppMessage('warning', 'Unavailable');

  expect(useNotificationsStore.getState().items).toHaveLength(2);
  expect(useNotificationsStore.getState().items[0]).toMatchObject({ id: appId, logger: 'app', read: false });
  expect(useNotificationsStore.getState().items[1]).toMatchObject({ logger: 'pipeline', read: true });
});
