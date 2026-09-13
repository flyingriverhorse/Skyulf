import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Node } from '@xyflow/react';
import { ValidationNavigation } from '../../../components/shared/ValidationField';
import { registry } from '../../../core/registry/NodeRegistry';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useViewStore } from '../../../core/store/useViewStore';
import { GeoDistanceNode, type GeoDistanceConfig } from './GeoDistanceNode';
import GeoDistanceSettings from './geodistance/GeoDistanceSettings';

const fixture = vi.hoisted(() => ({ wide: false, loading: false, missing: false }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({
  useDatasetSchema: (datasetId?: string) => ({
    data: datasetId && !fixture.missing ? { columns: {
      lat: { name: 'lat', dtype: 'Float64' }, lon: { name: 'lon', dtype: 'double' },
      destination_lat: { name: 'destination_lat', dtype: 'Int64' },
      destination_lon: { name: 'destination_lon', dtype: 'decimal(9,6)' },
      target: { name: 'target', dtype: 'int' }, removed: { name: 'removed', dtype: 'float' },
      city: { name: 'city', dtype: 'string' }, flag: { name: 'flag', dtype: 'boolean' },
    }, row_count: 2 } : undefined,
    isLoading: fixture.loading,
  }),
}));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({
  useIsWideContainer: () => [React.createRef<HTMLDivElement>(), fixture.wide],
}));

const validConfig: GeoDistanceConfig = {
  lat1_col: 'lat', lon1_col: 'lon', lat2_col: 'destination_lat', lon2_col: 'destination_lon',
  method: 'haversine', unit: 'km', output_column: '',
};
const coordinates = [
  ['lat1_col', 'Point 1 latitude', 'lat'], ['lon1_col', 'Point 1 longitude', 'lon'],
  ['lat2_col', 'Point 2 latitude', 'destination_lat'],
  ['lon2_col', 'Point 2 longitude', 'destination_lon'],
] as const;

/** Keep a target split and explicit drop upstream so selectors exercise real graph context. */
function graphNode(id: string, data: Record<string, unknown>): Node {
  return { id, position: { x: 0, y: 0 }, data };
}

/** Use controlled settings so consecutive edits test persisted config merges. */
function setup(overrides: Partial<GeoDistanceConfig> = {}) {
  const onChange = vi.fn();
  function Harness() {
    const [config, setConfig] = React.useState({ ...GeoDistanceNode.getDefaultConfig(), ...overrides });
    return <ValidationNavigation nodeId="geo"><GeoDistanceSettings nodeId="geo" config={config}
      onChange={(next) => { onChange(next); setConfig(next); }} /></ValidationNavigation>;
  }
  return { ...render(<Harness />), onChange };
}

beforeEach(() => {
  fixture.wide = false; fixture.loading = false; fixture.missing = false;
  registry.register(GeoDistanceNode);
  useViewStore.setState({ validationFocusRequest: null });
  useGraphStore.setState({
    nodes: [
      graphNode('dataset', { definitionType: 'dataset_node', datasetId: 'locations' }),
      graphNode('target', { definitionType: 'feature_target_split', target_column: 'target' }),
      graphNode('drop', { definitionType: 'drop_missing_columns', columns: ['removed'] }),
      graphNode('geo', { ...GeoDistanceNode.getDefaultConfig(), definitionType: 'GeoDistance' }),
    ],
    edges: [
      { id: 'a', source: 'dataset', target: 'target' }, { id: 'b', source: 'target', target: 'drop' },
      { id: 'c', source: 'drop', target: 'geo' },
    ],
    predictedSchemas: {}, executionResult: null,
  });
});

describe('GeoDistance coordinate settings', () => {
  it('shows loading feedback before the deferred settings become editable', async () => {
    // Opening the asynchronously loaded form must preserve its config and edit callback.
    const Settings = GeoDistanceNode.settings;
    const onChange = vi.fn();
    render(<Settings nodeId="geo" config={validConfig} onChange={onChange} />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading settings...');
    const latitude = await screen.findByRole('combobox', { name: 'Point 1 latitude' });
    expect(latitude).toHaveValue('lat');
    fireEvent.change(latitude, { target: { value: 'destination_lat' } });
    expect(onChange).toHaveBeenLastCalledWith({ ...validConfig, lat1_col: 'destination_lat' });
  });

  it.each([false, true])('edits every coordinate in wide=%s layout with numeric feature choices', (wide) => {
    // All four controls must merge independently and never offer a removed or target column.
    fixture.wide = wide;
    const { onChange } = setup();
    for (const [, label, value] of coordinates) {
      const select = screen.getByRole('combobox', { name: label });
      expect(within(select).getAllByRole('option').map(option => option.getAttribute('value')))
        .toEqual(['', 'lat', 'lon', 'destination_lat', 'destination_lon']);
      fireEvent.change(select, { target: { value } });
    }
    expect(onChange).toHaveBeenLastCalledWith(validConfig);
  });

  it('uses predicted upstream numeric columns instead of stale source columns', () => {
    // Generated features and upstream casts must be visible before execution.
    useGraphStore.setState({ predictedSchemas: {
      drop: { columns: ['new_lat', 'new_lon', 'target', 'removed', 'text'], dtypes: {
        new_lat: 'Float64', new_lon: 'float64', target: 'int64', removed: 'float64', text: 'String',
      } },
      geo: { columns: ['geo_distance_km'], dtypes: { geo_distance_km: 'float64' } },
    } });
    setup();
    const select = screen.getByRole('combobox', { name: 'Point 1 latitude' });
    expect(within(select).getAllByRole('option').map(option => option.getAttribute('value')))
      .toEqual(['', 'new_lat', 'new_lon']);
  });

  it('keeps an empty predicted schema authoritative', () => {
    // A known empty upstream result must not restore columns from the source dataset.
    useGraphStore.setState({ predictedSchemas: { drop: { columns: [], dtypes: {} } } });
    setup();
    expect(screen.getByText('No numeric columns available.')).toBeVisible();
    expect(within(screen.getByRole('combobox', { name: 'Point 1 latitude' })).getAllByRole('option'))
      .toHaveLength(1);
  });

  it('shows a stale selected column as unavailable without rewriting it', () => {
    // Stored references stay visible for correction after an upstream column is removed.
    const { onChange } = setup({ lat1_col: 'removed' });
    const option = screen.getByRole('option', { name: 'removed (unavailable)' });
    expect(option).toBeDisabled();
    expect(screen.getByRole('combobox', { name: 'Point 1 latitude' })).toHaveValue('removed');
    expect(onChange).not.toHaveBeenCalled();
  });

  it('keeps repeated coordinates valid for coincident points', () => {
    // Selecting the same point twice is a legitimate way to produce zero distance.
    expect(GeoDistanceNode.validate({ ...validConfig, lat2_col: 'lat', lon2_col: 'lon' })).toEqual({ isValid: true });
  });

  it('distinguishes missing dataset and loading schema states', () => {
    // Users need an actionable missing connection message before column choices load.
    useGraphStore.setState({ nodes: [], edges: [] });
    const { unmount } = setup();
    expect(screen.getByText('Connect a dataset node to see available columns.')).toBeVisible();
    unmount();
    fixture.loading = true;
    setup();
    expect(screen.getByText('Loading columns...')).toBeVisible();
  });
});

describe('GeoDistance method and output settings', () => {
  it('preserves coordinates while changing method, unit and the automatic output name', () => {
    // Both methods take degrees and unit changes must update only the automatic name.
    const { onChange } = setup(validConfig);
    expect(screen.getByText(/latitude and longitude in degrees/i)).toBeVisible();
    fireEvent.change(screen.getByRole('combobox', { name: 'Distance method' }), { target: { value: 'euclidean' } });
    expect(screen.getByText(/nearby points/i)).toBeVisible();
    fireEvent.change(screen.getByRole('combobox', { name: 'Distance unit' }), { target: { value: 'mi' } });
    expect(screen.getByRole('textbox', { name: 'Output column (optional)' })).toHaveAttribute('placeholder', 'geo_distance_mi');
    expect(onChange).toHaveBeenLastCalledWith({ ...validConfig, method: 'euclidean', unit: 'mi' });
  });

  it('retains an explicit output column across unit changes and restores auto naming when cleared', () => {
    // A user supplied feature name must survive edits to the measurement unit.
    const { onChange } = setup(validConfig);
    const output = screen.getByRole('textbox', { name: 'Output column (optional)' });
    fireEvent.change(output, { target: { value: 'journey_distance' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Distance unit' }), { target: { value: 'mi' } });
    expect(output).toHaveValue('journey_distance');
    fireEvent.change(output, { target: { value: '' } });
    expect(output).toHaveAttribute('placeholder', 'geo_distance_mi');
    expect(onChange).toHaveBeenLastCalledWith({ ...validConfig, unit: 'mi', output_column: '' });
  });

  it('summarizes the selected method and output without treating blank output as invalid', () => {
    // Compact node text must identify automatic versus explicitly named output features.
    expect(GeoDistanceNode.validate(validConfig)).toEqual({ isValid: true });
    expect(GeoDistanceNode.bodyPreview?.(validConfig)).toContain('geo_distance_km');
    expect(GeoDistanceNode.bodyPreview?.({ ...validConfig, unit: 'mi' })).toContain('geo_distance_mi');
    expect(GeoDistanceNode.bodyPreview?.({ ...validConfig, output_column: 'route' })).toContain('route');
  });
});

describe('GeoDistance validation navigation', () => {
  it.each(coordinates)('anchors an invalid %s to its actual control', async (field, label) => {
    // Configuration issues must focus the corresponding coordinate rather than the panel summary.
    const config = { ...validConfig, [field]: '' };
    useGraphStore.setState({ nodes: useGraphStore.getState().nodes.map(node => node.id === 'geo'
      ? { ...node, data: { ...config, definitionType: 'GeoDistance' } } : node) });
    setup(config);
    expect(GeoDistanceNode.validate(config)).toMatchObject({ isValid: false, field });
    act(() => useViewStore.getState().requestValidationFocus({
      nodeId: 'geo', nodeLabel: 'Geo Distance', category: 'configuration', field, message: 'Select a column.',
    }));
    await waitFor(() => expect(screen.getByRole('combobox', { name: label })).toHaveFocus());
    expect(screen.getByRole('combobox', { name: label })).toHaveAttribute('aria-invalid', 'true');
  });

  it.each([
    ['lat1_col', '   '], ['lon1_col', null], ['lat2_col', 0], ['lon2_col', undefined],
    ['method', 'manhattan'], ['unit', 'm'], ['output_column', 42],
  ])('rejects malformed imported %s values', (field, value) => {
    // Imported canvas JSON must not bypass the required coordinates or supported enum contract.
    expect(GeoDistanceNode.validate({ ...validConfig, [field]: value } as GeoDistanceConfig))
      .toMatchObject({ isValid: false, field });
  });

  it.each(['method', 'unit', 'output_column'])('renders a validation anchor for %s', (field) => {
    // Every non-coordinate validator result must also have an editable field target.
    const { container } = setup();
    expect(container.querySelector(`[data-validation-field="${field}"]`)?.querySelector('input, select'))
      .toBeInTheDocument();
  });
});
