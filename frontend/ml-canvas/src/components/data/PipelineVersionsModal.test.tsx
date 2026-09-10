import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { pipelineVersionsApi, type PipelineVersionEntry } from '../../core/api/pipelineVersions';
import { ConfirmProvider } from '../shared';
import { PipelineVersionsModal } from './PipelineVersionsModal';

afterEach(() => vi.restoreAllMocks());
const entry: PipelineVersionEntry = { id: 7, datasetId: 'ds', versionInt: 3, name: 'Snapshot', kind: 'auto', pinned: true, nodeCount: 4, edgeCount: 0, createdAt: '2026-01-01', graph: { nodes: [{ type: 'fallback', data: { definitionType: '', catalogType: 'ignored' } }, { data: { catalogType: 'Scale' } }, { data: { definitionType: 'Scale' } }, null] } };

function mountVersions() {
  /** Keep real modal and confirmation behavior while isolating the API. */
  const onRestore = vi.fn();
  const onClose = vi.fn();
  const result = render(<ConfirmProvider><PipelineVersionsModal isOpen onClose={onClose} datasetId="ds" datasetName="Data" onRestore={onRestore} /></ConfirmProvider>);
  return { ...result, onRestore, onClose };
}

describe('PipelineVersionsModal public actions', () => {
  it('keeps server order, graph fallback labels and restore confirmation', async () => {
    /** Restoring must hand off the original snapshot only after confirmation. */
    vi.spyOn(pipelineVersionsApi, 'list').mockResolvedValue([entry, { ...entry, id: 8, name: 'Second', pinned: false, graph: [{ step_type: 'Train' }, {}, null] }]);
    const { onRestore, onClose } = mountVersions();
    await screen.findByText('Snapshot');
    expect(screen.getAllByTitle(/^(Snapshot|Second)$/).map(node => node.textContent)).toEqual(['Snapshot', 'Second']);
    fireEvent.click(screen.getAllByRole('button', { name: 'Toggle version details' })[0]!);
    expect(screen.getByTitle('Scale')).toHaveTextContent('×2');
    expect(screen.getAllByText('4 nodes · 0 edges')).toHaveLength(2);
    fireEvent.click(screen.getAllByRole('button', { name: 'Restore version' })[0]!);
    expect(onRestore).not.toHaveBeenCalled();
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Restore v3 "Snapshot"?' })).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Restore v3 "Snapshot"?' })).not.toBeInTheDocument());
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.click(screen.getAllByRole('button', { name: 'Restore version' })[0]!);
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Restore v3 "Snapshot"?' })).getByRole('button', { name: 'Restore' }));
    await waitFor(() => expect(onRestore).toHaveBeenCalledWith(entry));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('preserves edit drafts, omits unchanged names and clears whitespace notes', async () => {
    /** Inline editing sends only a changed name but always sends the note. */
    vi.spyOn(pipelineVersionsApi, 'list').mockResolvedValue([entry]);
    const update = vi.spyOn(pipelineVersionsApi, 'update').mockResolvedValue(entry);
    mountVersions();
    fireEvent.click(await screen.findByRole('button', { name: 'Edit version' }));
    fireEvent.change(screen.getByLabelText('Pipeline name'), { target: { value: ' Snapshot ' } });
    fireEvent.change(screen.getByLabelText('Version note'), { target: { value: '   ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(update).toHaveBeenCalledWith('ds', 7, { note: null }));
    await waitFor(() => expect(screen.queryByLabelText('Pipeline name')).not.toBeInTheDocument());
  });

  it('renders engine graph counts and empty graph messaging on expansion', async () => {
    /** Both persisted graph formats retain their public breakdowns. */
    vi.spyOn(pipelineVersionsApi, 'list').mockResolvedValue([{ ...entry, graph: [{ step_type: 'Train' }, {}, null] }, { ...entry, id: 8, name: 'Empty', graph: null }]);
    mountVersions();
    fireEvent.click((await screen.findAllByRole('button', { name: 'Toggle version details' }))[0]!);
    expect(screen.getByTitle('Train')).toHaveTextContent('×1');
    expect(screen.getByTitle('unknown')).toHaveTextContent('×1');
    fireEvent.click(screen.getAllByRole('button', { name: 'Toggle version details' })[1]!);
    expect(screen.getByText('Snapshot graph is empty or in an unrecognised shape.')).toBeInTheDocument();
  });
});
