import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { EncodingNode } from './EncodingNode';

const Settings = EncodingNode.settings!;

const renderSettings = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Settings config={EncodingNode.getDefaultConfig()} onChange={() => {}} />
    </QueryClientProvider>,
  );
};

describe('EncodingNode settings accessible names', () => {
  it('associates the visible "Encoding Method" label with its select', () => {
    renderSettings();

    const select = screen.getByRole('combobox', { name: 'Encoding Method' });
    expect(select).toBeInTheDocument();
    expect(select.id).not.toBe('');
  });
});
