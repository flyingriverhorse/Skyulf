import { describe, expect, it, beforeEach, beforeAll } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import userEvent from '@testing-library/user-event';
import { PropertiesPanel } from './PropertiesPanel';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { registry } from '../../core/registry/NodeRegistry';
import { initializeRegistry } from '../../core/registry/init';
import { ValidationField } from '../shared/ValidationField';
import { useNodeInspectionStore } from '../../core/store/useNodeInspectionStore';

const MERGE_NODE = {
  id: 'merge-node',
  type: 'custom',
  position: { x: 0, y: 0 },
  selected: true,
  data: { definitionType: 'MissingIndicator', label: 'MissingIndicator', columns: [] },
};

const EDGES = [
  { id: 'e1', source: 'branch-a', target: 'merge-node' },
  { id: 'e2', source: 'branch-b', target: 'merge-node' },
];

/** Seed the store with a two-branch fan-in and an optional engine advisory. */
const seedStore = (mergeWarnings: unknown[]) => {
  useGraphStore.setState({
    nodes: [MERGE_NODE],
    edges: EDGES,
    executionResult: mergeWarnings.length
      ? ({ merge_warnings: mergeWarnings, node_results: {} } as never)
      : null,
  } as never);
};

describe('PropertiesPanel merge strategy', () => {
  beforeAll(() => initializeRegistry());

  const renderPanel = () =>
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PropertiesPanel />
      </QueryClientProvider>,
    );

  beforeEach(() => {
    useGraphStore.setState({ executionResult: null } as never);
  });

  it('keeps a single editor when switching between nodes repeatedly', () => {
    // Duplicate sibling keys must not leave old technical-details blocks in the next editor.
    useGraphStore.setState({ nodes: [MERGE_NODE, { ...MERGE_NODE, id: 'second-node', selected: false }], edges: [] });
    const { container } = renderPanel();
    for (const nodeId of ['second-node', 'merge-node', 'second-node', 'merge-node']) {
      act(() => { useGraphStore.getState().selectNode(nodeId); });
      expect(container.querySelectorAll('details').length).toBeLessThanOrEqual(1);
      expect(screen.getAllByRole('button', { name: 'Node information' })).toHaveLength(1);
    }
    expect(screen.getAllByRole('heading', { name: 'MissingIndicator' })).toHaveLength(1);
  });

  it('hides the strategy control when no branch contested a column', () => {
    seedStore([]);
    renderPanel();
    expect(screen.queryByText('Merge Strategy')).not.toBeInTheDocument();
  });

  it('predicts a conflict before any run when both branches target the same column', () => {
    useGraphStore.setState({
      nodes: [
        MERGE_NODE,
        { id: 'branch-a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'TransformationNode', label: 'TransformationNode', transformations: [{ columns: ['SepalLengthCm'], method: 'log' }] } },
        { id: 'branch-b', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'TransformationNode', label: 'TransformationNode2', transformations: [{ columns: ['SepalLengthCm'], method: 'cube' }] } },
      ],
      edges: EDGES,
      executionResult: null,
    } as never);

    renderPanel();
    expect(screen.getByText('Merge Strategy')).toBeTruthy();
    expect(screen.getByText('Predicted')).toBeTruthy();
    expect(screen.getByText(/SepalLengthCm/)).toBeTruthy();
  });

  it('hides it even when a run produced only an upstream-drop advisory', () => {
    seedStore([
      { node_id: 'merge-node', kind: 'upstream_drop_reapplied', dropped_columns: ['Id'] },
    ]);
    renderPanel();
    expect(screen.queryByText('Merge Strategy')).not.toBeInTheDocument();
  });

  it('names the merge strategy control and applies the selected winner', () => {
    // Assistive technology must identify which setting changes overlapping-column ownership.
    seedStore([{ node_id: 'merge-node', kind: 'sibling_fan_in', inputs: ['branch-a', 'branch-b'], overlap_columns: ['age'] }]);
    renderPanel();
    const strategy = screen.getByRole('combobox', { name: 'Merge Strategy' });
    fireEvent.change(strategy, { target: { value: 'first_wins' } });
    expect(useGraphStore.getState().nodes.find(node => node.id === 'merge-node')?.data.merge_strategy).toBe('first_wins');
  });

  it('shows it, naming the contested columns, once two branches edited the same column', () => {
    useGraphStore.setState({
      nodes: [
        MERGE_NODE,
        { id: 'branch-a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'drop_columns', label: 'Drop Missing Columns' } },
        { id: 'branch-b', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'SimpleTransformation', label: 'TransformationNode' } },
      ],
      edges: EDGES,
      executionResult: {
        node_results: {},
        merge_warnings: [
          {
            node_id: 'merge-node',
            kind: 'sibling_fan_in',
            inputs: ['branch-a', 'branch-b'],
            overlap_columns: ['SepalLengthCm'],
            winner_input: 'branch-b',
          },
        ],
      },
    } as never);

    renderPanel();
    expect(screen.getByText('Merge Strategy')).toBeTruthy();
    expect(screen.getByText(/SepalLengthCm/)).toBeTruthy();
    expect(screen.getByText('Drop Missing Columns')).toBeTruthy();
    expect(screen.getByText('TransformationNode')).toBeTruthy();
    expect(screen.getByText(/Keep TransformationNode \(last connected/)).toBeTruthy();
    expect(screen.getByText(/Keep Drop Missing Columns \(first connected/)).toBeTruthy();
  });
});

describe('PropertiesPanel multi-input mode merge-winner hint', () => {
  beforeAll(() => initializeRegistry());

  const TRAINING_NODE = {
    id: 'train-node',
    type: 'custom',
    position: { x: 0, y: 0 },
    selected: true,
    data: {
      definitionType: 'classification',
      label: 'Training',
      // ClassificationNode defaultConfig — TrainingSettings reads these fields.
      run_mode: 'basic',
      model_type: 'random_forest_classifier',
      hyperparameters: {},
      cv_enabled: true,
      cv_folds: 5,
      cv_type: 'k_fold',
      cv_shuffle: true,
      cv_random_state: 42,
      cv_time_column: '',
      n_trials: 10,
      metric: 'accuracy',
      search_strategy: 'random',
      random_state: 42,
      search_space: {},
    },
  };

  const seedTraining = (executionMode?: string) => {
    useGraphStore.setState({
      nodes: [
        { ...TRAINING_NODE, data: { ...TRAINING_NODE.data, ...(executionMode ? { execution_mode: executionMode } : {}) } },
        { id: 'branch-a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'WOEEncoder', label: 'Encoder' } },
        { id: 'branch-b', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'Scale', label: 'Scaler' } },
      ],
      edges: [
        { id: 'e1', source: 'branch-a', target: 'train-node' },
        { id: 'e2', source: 'branch-b', target: 'train-node' },
      ],
      executionResult: null,
    } as never);
  };

  const renderPanel = () =>
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PropertiesPanel />
      </QueryClientProvider>,
    );

  it('in merge mode names the last connected branch as the winner', () => {
    seedTraining('merge');
    renderPanel();
    expect(screen.getByText(/If two branches carry the same column/)).toBeTruthy();
    expect(screen.getByText('Scaler')).toBeTruthy();
  });

  it('in parallel mode shows no merge-winner hint', () => {
    seedTraining('parallel');
    renderPanel();
    expect(screen.queryByText(/If two branches carry the same column/)).not.toBeInTheDocument();
  });

  it('announces the selected multi-input mode when toggling it', () => {
    // Color alone must not be the only way to distinguish merged and parallel execution.
    seedTraining('merge');
    renderPanel();
    const merge = screen.getByRole('button', { name: 'Merge' });
    const parallel = screen.getByRole('button', { name: 'Parallel' });
    expect(merge).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(parallel);
    expect(merge).toHaveAttribute('aria-pressed', 'false');
    expect(parallel).toHaveAttribute('aria-pressed', 'true');
  });
});

/** Keep an unsaved local edit so remounting settings loses observable state. */
function InspectionTestSettings() {
  const [draft, setDraft] = useState('');
  return <ValidationField field="name"><input aria-label="Draft name" value={draft}
    onChange={(event) => setDraft(event.target.value)} /></ValidationField>;
}

describe('PropertiesPanel inspection tabs', () => {
  beforeAll(() => {
    registry.register({
      type: 'inspection_tabs_test', label: 'Inspection test', category: 'Utility',
      description: '', inputs: [], outputs: [], settings: InspectionTestSettings,
      getDefaultConfig: () => ({}),
      validate: () => ({ isValid: false, field: 'name', message: 'Enter a name' }),
    });
  });

  beforeEach(() => {
    useGraphStore.setState({
      nodes: [
        { id: 'inspect-first', position: { x: 0, y: 0 }, data: { definitionType: 'inspection_tabs_test' }, selected: true },
        { id: 'inspect-second', position: { x: 0, y: 0 }, data: { definitionType: 'inspection_tabs_test' } },
      ], edges: [], executionResult: null,
    });
    useViewStore.setState({ validationFocusRequest: null });
    useNodeInspectionStore.setState({ receipt: null, isLoading: false, error: null });
    useViewStore.setState({ isPropertiesPanelExpanded: false, propertiesPanelWidth: 400 });
  });

  it('keeps resize bounds and clears expansion and content when selection closes', () => {
    // Collapsing and deselecting must retain the preferred width without leaving an editor visible.
    render(<PropertiesPanel />);
    const handle = screen.getByRole('separator', { name: 'Resize settings panel' });
    expect(handle).toHaveAttribute('aria-valuemin', '320');
    // The default ResizeObserver shim reports no width, so the reserved canvas bounds clamp to 320.
    expect(handle).toHaveAttribute('aria-valuemax', '320');
    fireEvent.keyDown(handle, { key: 'End' });
    expect(useViewStore.getState().propertiesPanelWidth).toBe(320);
    fireEvent.click(screen.getByRole('button', { name: 'Expand settings panel' }));
    expect(screen.queryByRole('separator')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close settings panel' }));
    expect(useViewStore.getState().isPropertiesPanelExpanded).toBe(false);
    expect(screen.queryByRole('tab')).not.toBeInTheDocument();
    expect(screen.getByRole('complementary', { name: 'Node settings' })).toHaveClass('w-0', 'opacity-0');
  });

  it('keeps settings edits mounted while inspecting input and output', async () => {
    // Inspecting a sample must not discard an in-progress settings form.
    const user = userEvent.setup();
    render(<PropertiesPanel />);
    const draft = screen.getByRole('textbox', { name: 'Draft name' });
    await act(async () => { await user.type(draft, 'unfinished'); });
    await act(async () => { await user.click(screen.getByRole('tab', { name: 'Input' })); });
    expect(draft).toBeInTheDocument();
    expect(draft).not.toBeVisible();
    expect(screen.getByRole('tabpanel', { name: 'Input' })).toBeVisible();
    await act(async () => { await user.click(screen.getByRole('tab', { name: 'Output' })); });
    expect(screen.getByRole('tabpanel', { name: 'Output' })).toBeVisible();
    await act(async () => { await user.click(screen.getByRole('tab', { name: 'Settings' })); });
    expect(screen.getByRole('textbox', { name: 'Draft name' })).toHaveValue('unfinished');
  });

  it('supports arrow keys and preserves tab focus when selection changes', async () => {
    // Node selection must not remount the focused tab or strand keyboard navigation.
    const user = userEvent.setup();
    render(<PropertiesPanel />);
    const settings = screen.getByRole('tab', { name: 'Settings' });
    act(() => settings.focus());
    await act(async () => { await user.keyboard('{ArrowRight}'); });
    expect(screen.getByRole('tab', { name: 'Input' })).toHaveFocus();
    expect(screen.getByRole('tab', { name: 'Input' })).toHaveAttribute('aria-selected', 'true');
    await act(async () => { await user.keyboard('{End}'); });
    const output = screen.getByRole('tab', { name: 'Output' });
    expect(output).toHaveFocus();
    act(() => useGraphStore.getState().selectNode('inspect-second'));
    expect(output).toHaveFocus();
    expect(output).toHaveAttribute('aria-selected', 'true');
    await act(async () => { await user.keyboard('{ArrowRight}'); });
    expect(settings).toHaveFocus();
    await act(async () => { await user.keyboard('{ArrowLeft}'); });
    expect(output).toHaveFocus();
    await act(async () => { await user.keyboard('{Home}'); });
    expect(settings).toHaveFocus();
  });

  it('preserves the selected branch when comparing Input and Output tabs', () => {
    // Tab switches must compare measurements from the same branch execution.
    useNodeInspectionStore.setState({ receipt: {
      configurationKey: 'captured-configuration',
      response: { pipeline_id: 'pipeline', status: 'success', node_results: {}, preview_data: null,
        recommendations: [], run_id: 'receipt', node_inspections:
        ['Branch A', 'Branch B'].map((label, index) => ({
          node_id: 'inspect-first', branch_id: `branch-${index}`, branch_label: label,
          input: { status: 'available' as const, reason: null, tables: [] },
          output: { status: 'available' as const, reason: null, tables: [] },
        })),
      },
    } });
    render(<PropertiesPanel />);
    fireEvent.click(screen.getByRole('tab', { name: 'Output' }));
    fireEvent.change(screen.getByRole('combobox', { name: 'Data path' }), { target: { value: 'branch-1' } });
    fireEvent.click(screen.getByRole('tab', { name: 'Input' }));
    expect(screen.getByRole('combobox', { name: 'Data path' })).toHaveDisplayValue('Branch B');
    fireEvent.click(screen.getByRole('tab', { name: 'Output' }));
    expect(screen.getByRole('combobox', { name: 'Data path' })).toHaveDisplayValue('Branch B');
  });

  it('reveals Settings and focuses a validation field hidden by inspection', async () => {
    // An issue link must reach the editable control even while samples are open.
    const user = userEvent.setup();
    render(<PropertiesPanel />);
    await act(async () => { await user.click(screen.getByRole('tab', { name: 'Output' })); });
    act(() => useViewStore.getState().requestValidationFocus({
      nodeId: 'inspect-first', nodeLabel: 'Inspection test', category: 'configuration',
      field: 'name', message: 'Enter a name',
    }));
    const input = await screen.findByRole('textbox', { name: 'Draft name' });
    expect(screen.getByRole('tab', { name: 'Settings' })).toHaveAttribute('aria-selected', 'true');
    expect(input).toHaveFocus();
    await act(async () => { await user.click(screen.getByRole('tab', { name: 'Input' })); });
    expect(screen.getByRole('tab', { name: 'Input' })).toHaveFocus();
    expect(screen.getByRole('tab', { name: 'Input' })).toHaveAttribute('aria-selected', 'true');
  });
});
