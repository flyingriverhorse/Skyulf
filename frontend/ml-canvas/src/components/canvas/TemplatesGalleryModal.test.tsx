import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { initializeRegistry } from '../../core/registry/init';
import { useGraphStore } from '../../core/store/useGraphStore';
import { TemplatesGalleryModal } from './TemplatesGalleryModal';

const boundary = vi.hoisted(() => ({ confirm: vi.fn(), refetch: vi.fn(), schemaError: false }));
vi.mock('../shared', () => ({ useConfirm: () => boundary.confirm }));
vi.mock('../../core/toast', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));
vi.mock('../../core/hooks/useDatasets', () => ({ useUsableDatasets: () => ({
  data: [{ id: '1', name: 'Customers' }, { id: '2', name: 'Other data' }], isLoading: false, isError: false,
}) }));
vi.mock('../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: () => ({
  data: { columns: { label: { name: 'label', dtype: 'string' }, message: { name: 'message', dtype: 'string' }, price: { name: 'price', dtype: 'float64' } } },
  isLoading: false, isError: boundary.schemaError, refetch: boundary.refetch,
}) }));

beforeAll(() => initializeRegistry());
beforeEach(() => {
  boundary.schemaError = false;
  boundary.confirm.mockResolvedValue(true);
  useGraphStore.setState({ nodes: [], edges: [] });
});

describe('template setup', () => {
  it('requires data and target, then opens the configured graph', async () => {
    // Setup must bind the chosen source and target rather than leave an empty starter.
    const close = vi.fn();
    render(<TemplatesGalleryModal isOpen onClose={close} />);
    fireEvent.click(screen.getByTestId('template-card-tabular_classification'));
    expect(screen.getByRole('button', { name: 'Open in Canvas' })).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Dataset'), { target: { value: '1' } });
    fireEvent.change(screen.getByLabelText('Target column'), { target: { value: 'label' } });
    fireEvent.click(screen.getByRole('button', { name: 'Open in Canvas' }));
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    expect(useGraphStore.getState().nodes.find(n => n.data.definitionType === 'TrainTestSplitter')?.data.target_column).toBe('label');
  });

  it('clears selected columns when the dataset changes', () => {
    // A column selected on one dataset cannot silently carry into another.
    render(<TemplatesGalleryModal isOpen onClose={vi.fn()} />);
    fireEvent.click(screen.getByTestId('template-card-tabular_classification'));
    fireEvent.change(screen.getByLabelText('Dataset'), { target: { value: '1' } });
    fireEvent.change(screen.getByLabelText('Target column'), { target: { value: 'label' } });
    fireEvent.change(screen.getByLabelText('Dataset'), { target: { value: '2' } });
    expect(screen.getByLabelText('Target column')).toHaveValue('');
    expect(screen.getByRole('button', { name: 'Open in Canvas' })).toBeDisabled();
  });

  it('does not ask for a target for segmentation', () => {
    // Unsupervised workflows must remain usable without a label column.
    render(<TemplatesGalleryModal isOpen onClose={vi.fn()} />);
    fireEvent.click(screen.getByTestId('template-card-customer_segmentation'));
    fireEvent.change(screen.getByLabelText('Dataset'), { target: { value: '1' } });
    expect(screen.queryByLabelText('Target column')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open in Canvas' })).toBeEnabled();
  });

  it('keeps the current graph when replacement is declined', async () => {
    // Exploring a starter must not discard the existing canvas without approval.
    boundary.confirm.mockResolvedValue(false);
    const existing = [{ id: 'existing', position: { x: 0, y: 0 }, data: {} }];
    useGraphStore.setState({ nodes: existing });
    render(<TemplatesGalleryModal isOpen onClose={vi.fn()} />);
    fireEvent.click(screen.getByTestId('template-card-customer_segmentation'));
    fireEvent.change(screen.getByLabelText('Dataset'), { target: { value: '1' } });
    fireEvent.click(screen.getByRole('button', { name: 'Open in Canvas' }));
    await waitFor(() => expect(boundary.confirm).toHaveBeenCalled());
    expect(useGraphStore.getState().nodes).toEqual(existing);
  });

  it('blocks setup when the schema request fails', () => {
    // Cached columns must not hide a failed schema lookup.
    boundary.schemaError = true;
    render(<TemplatesGalleryModal isOpen onClose={vi.fn()} />);
    fireEvent.click(screen.getByTestId('template-card-tabular_classification'));
    fireEvent.change(screen.getByLabelText('Dataset'), { target: { value: '1' } });
    expect(screen.getByRole('alert')).toHaveTextContent('Could not load columns');
    expect(screen.getByRole('button', { name: 'Open in Canvas' })).toBeDisabled();
  });
});
