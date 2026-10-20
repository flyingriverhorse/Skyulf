import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AxiosError, type AxiosResponse } from 'axios';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../../core/api/client';
import { ExportProjectButton } from './ExportProjectButton';

const props = { jobId: 'job-older', modelName: 'Random Forest', version: 2 };

function blobError(content: string): AxiosError<Blob> {
  const data = new Blob([content], { type: 'application/json' });
  // jsdom's Blob lacks text(); browsers provide it on binary Axios responses.
  Object.defineProperty(data, 'text', { value: async () => content });
  return new AxiosError('Request failed with status code 400', 'ERR_BAD_REQUEST', undefined, undefined, {
    data, status: 400, statusText: 'Bad Request', headers: {}, config: {},
  } as AxiosResponse<Blob>);
}

describe('ExportProjectButton', () => {
  const createObjectURL = vi.fn(() => 'blob:project-download');
  const revokeObjectURL = vi.fn();
  let downloads: { filename: string; href: string }[];

  beforeEach(() => {
    downloads = [];
    vi.stubGlobal('URL', class extends URL {
      static createObjectURL = createObjectURL;
      static revokeObjectURL = revokeObjectURL;
    });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      downloads.push({ filename: this.download, href: this.href });
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it.each(['both', 'api', 'batch'])('downloads the selected job with %s mode and releases its blob URL', async (mode) => {
    // The actual API client must receive the chosen version and archive format.
    const archive = new Blob(['zip contents'], { type: 'application/zip' });
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: archive });
    render(<ExportProjectButton {...props} />);
    const select = screen.getByRole('combobox', { name: 'Project format for Random Forest v2' });
    expect(select).toHaveValue('both');
    await act(async () => { await userEvent.selectOptions(select, mode); });
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Export project for Random Forest v2' })); });

    expect(post).toHaveBeenCalledWith('/deployment/export/job-older', undefined, {
      params: { mode }, responseType: 'blob',
    });
    expect(downloads).toEqual([{ filename: 'kayra_model_job-older.zip', href: 'blob:project-download' }]);
    expect(createObjectURL).toHaveBeenCalledWith(archive);
    await waitFor(() => { expect(revokeObjectURL).toHaveBeenCalledWith('blob:project-download'); });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('encodes job identity for the request and sanitizes the download filename', async () => {
    // Special characters must not change the request path or create an unsafe filename.
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: new Blob(['zip']) });
    render(<ExportProjectButton {...props} jobId="job ?#7" />);
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: /Export project/ })); });

    expect(post).toHaveBeenCalledWith('/deployment/export/job%20%3F%237', undefined, {
      params: { mode: 'both' }, responseType: 'blob',
    });
    expect(downloads[0]?.filename).toBe('kayra_model_job___7.zip');
  });

  it('keeps busy state local to one version and prevents duplicate submissions', async () => {
    // A slow export must not lock another model version or create duplicate ZIP requests.
    let finish!: (value: { data: Blob }) => void;
    const pending = new Promise<{ data: Blob }>((resolve) => { finish = resolve; });
    const post = vi.spyOn(apiClient, 'post').mockReturnValue(pending);
    render(<><ExportProjectButton {...props} /><ExportProjectButton {...props} jobId="job-latest" version={3} /></>);
    const button = screen.getByRole('button', { name: 'Export project for Random Forest v2' });
    fireEvent.click(button);
    fireEvent.click(button);

    expect(button).toBeDisabled();
    expect(button).toHaveAttribute('aria-busy', 'true');
    expect(screen.getByRole('combobox', { name: 'Project format for Random Forest v2' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Export project for Random Forest v3' })).toBeEnabled();
    expect(post).toHaveBeenCalledTimes(1);
    await act(async () => { finish({ data: new Blob(['zip']) }); });
    expect(button).toBeEnabled();
  });

  it('shows a binary JSON rejection and allows retry without retaining the error', async () => {
    // Backend unsupported-artifact details must survive Axios's binary response mode.
    vi.spyOn(apiClient, 'post')
      .mockRejectedValueOnce(blobError('{"detail":"LagFeatures needs historical context. Export is unavailable."}'))
      .mockResolvedValueOnce({ data: new Blob(['zip']) });
    render(<ExportProjectButton {...props} />);
    const button = screen.getByRole('button', { name: /Export project/ });
    await act(async () => { await userEvent.click(button); });

    expect(await screen.findByRole('alert')).toHaveTextContent('LagFeatures needs historical context. Export is unavailable.');
    expect(downloads).toHaveLength(0);
    expect(button).toBeEnabled();
    await act(async () => { await userEvent.click(button); });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(downloads).toHaveLength(1);
  });

  it.each([
    {
      name: 'the backend HTTP exception envelope',
      body: {
        success: false,
        error: 'HTTP 400',
        message: 'Raw input schema is unavailable. Retrain this model before exporting.',
        status_code: 400,
      },
      expected: 'Raw input schema is unavailable. Retrain this model before exporting.',
    },
    {
      name: 'detail before message when both are strings',
      body: { detail: 'Export requires historical context.', message: 'Export failed.' },
      expected: 'Export requires historical context.',
    },
    {
      name: 'message when detail is structured',
      body: { detail: [{ msg: 'invalid mode' }], message: 'Choose API, batch, or both.' },
      expected: 'Choose API, batch, or both.',
    },
  ])('shows $name from a binary rejection', async ({ body, expected }) => {
    // Both backend envelopes must retain actionable text without rendering structured fields.
    vi.spyOn(apiClient, 'post').mockRejectedValueOnce(blobError(JSON.stringify(body)));
    render(<ExportProjectButton {...props} />);
    const button = screen.getByRole('button', { name: /Export project/ });
    await act(async () => { await userEvent.click(button); });
    expect(await screen.findByRole('alert')).toHaveTextContent(expected);
    expect(button).toBeEnabled();
    expect(downloads).toHaveLength(0);
  });

  it.each([
    '<html>Gateway error</html>',
    '{"message":',
    '{"detail":[{"msg":"invalid mode"}]}',
    '{"message":["invalid mode"]}',
    '{"detail":400,"message":{"reason":"invalid mode"}}',
    'null',
  ])(
    'preserves a usable error when a binary response has no string detail or message',
    async (body) => {
      // Proxy HTML and structured validation responses must not leak JSON parse failures.
      vi.spyOn(apiClient, 'post').mockRejectedValueOnce(blobError(body));
      render(<ExportProjectButton {...props} />);
      await act(async () => { await userEvent.click(screen.getByRole('button', { name: /Export project/ })); });
      expect(await screen.findByRole('alert')).toHaveTextContent('Request failed with status code 400');
      expect(downloads).toHaveLength(0);
    },
  );

  it('shows a network failure and releases busy state', async () => {
    // A failed connection must leave the control available for retry.
    vi.spyOn(apiClient, 'post').mockRejectedValueOnce(new Error('Network error — please try again.'));
    render(<ExportProjectButton {...props} />);
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: /Export project/ })); });
    expect(await screen.findByRole('alert')).toHaveTextContent('Network error');
    expect(screen.getByRole('button', { name: /Export project/ })).toBeEnabled();
  });
});
