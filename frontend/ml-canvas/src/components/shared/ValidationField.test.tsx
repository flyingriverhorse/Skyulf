import { act, fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { useState } from 'react';
import { registry } from '../../core/registry/NodeRegistry';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { ValidationField, ValidationNavigation, useValidationReveal } from './ValidationField';

const issue = { nodeId: 'invalid', nodeLabel: 'Example', category: 'configuration' as const, field: 'name', message: 'Enter a name' };

function Settings() {
  const [open, setOpen] = useState(false);
  useValidationReveal(() => setOpen(true));
  return <><button onClick={() => setOpen(false)}>Hide</button>{open && <ValidationField field="name"><input aria-label="Name" onChange={(event) => useGraphStore.getState().updateNodeData('invalid', { name: event.target.value })} /></ValidationField>}</>;
}

describe('validation navigation', () => {
  beforeEach(() => {
    registry.register({ type: 'validation_test', label: 'Example', category: 'Utility', description: '', inputs: [], outputs: [], settings: Settings, getDefaultConfig: () => ({}), validate: (config: { name?: string }) => config.name ? { isValid: true } : { isValid: false, field: 'name', message: 'Enter a name' } });
    useGraphStore.setState({ nodes: [{ id: 'invalid', position: { x: 0, y: 0 }, data: { definitionType: 'validation_test' }, selected: true }], edges: [] });
    useViewStore.setState({ validationFocusRequest: null });
  });

  it('reveals hidden fields, focuses them, and clears errors without moving focus on correction', async () => {
    // A correction must not interrupt typing or keep an obsolete error attached.
    render(<ValidationNavigation nodeId="invalid"><Settings /></ValidationNavigation>);
    act(() => useViewStore.getState().requestValidationFocus(issue));
    const input = await screen.findByRole('textbox', { name: 'Name' });
    expect(input).toHaveFocus();
    expect(input).toHaveAttribute('aria-invalid', 'true');
    expect(input).toHaveAccessibleDescription(/Enter a name/);
    fireEvent.change(input, { target: { value: 'fixed' } });
    expect(input).toHaveFocus();
    expect(input).not.toHaveAttribute('aria-invalid');
  });

  it('repeats navigation to the same issue and falls back to a summary for general issues', async () => {
    // Activating an issue again must reopen a section the user closed.
    render(<ValidationNavigation nodeId="invalid"><Settings /></ValidationNavigation>);
    act(() => useViewStore.getState().requestValidationFocus(issue));
    fireEvent.click(screen.getByRole('button', { name: 'Hide' }));
    act(() => useViewStore.getState().requestValidationFocus(issue));
    expect(await screen.findByRole('textbox', { name: 'Name' })).toHaveFocus();
    act(() => useViewStore.getState().requestValidationFocus({ ...issue, field: undefined }));
    expect(screen.getByRole('group', { name: 'Validation issue' })).toHaveFocus();
  });

  it('waits for asynchronous controls but cancels when the user moves elsewhere', async () => {
    // A late dataset response must never steal focus after the user continues working.
    const contents = (ready: boolean) => <ValidationNavigation nodeId="invalid"><button>Elsewhere</button><ValidationField field="name">{ready ? <input aria-label="Loaded name" /> : 'Loading'}</ValidationField></ValidationNavigation>;
    const { rerender } = render(contents(false));
    act(() => useViewStore.getState().requestValidationFocus(issue));
    rerender(contents(true));
    expect(await screen.findByRole('textbox', { name: 'Loaded name' })).toHaveFocus();
    rerender(contents(false));
    act(() => useViewStore.getState().requestValidationFocus(issue));
    act(() => screen.getByRole('button', { name: 'Elsewhere' }).focus());
    rerender(contents(true));
    expect(screen.getByRole('button', { name: 'Elsewhere' })).toHaveFocus();
  });
});
