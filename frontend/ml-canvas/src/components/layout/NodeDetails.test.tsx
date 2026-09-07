import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { NodeDetails } from './NodeDetails';

afterEach(() => { vi.unstubAllGlobals(); });

describe('node information', () => {
  it.each([true, false])('reports clipboard success=%s and keeps the identifier available', async success => {
    // A denied clipboard write must leave the ID readable and explain how to copy it manually.
    const writeText = success ? vi.fn().mockResolvedValue(undefined) : vi.fn().mockRejectedValue(new Error('Denied'));
    vi.stubGlobal('navigator', { clipboard: { writeText } });
    render(<NodeDetails nodeId="classification-example" />);
    fireEvent.click(screen.getByRole('button', { name: 'Node information' }));
    fireEvent.click(screen.getByRole('button', { name: 'Copy node ID' }));
    expect(writeText).toHaveBeenCalledWith('classification-example');
    expect(await screen.findByText(success ? 'Node ID copied.' : 'Could not copy. Select the ID to copy it manually.')).toBeVisible();
    expect(screen.getByText('classification-example')).toBeVisible();
  });
});
