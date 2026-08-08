import { describe, it, expect } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { RecordLink } from './RecordLink';
import { parseOperationalContext } from '../../core/utils/operationalContext';

function renderLink(ui: React.ReactElement) {
  return render(<MemoryRouter>{ui}</MemoryRouter>);
}

/** The href a rendered RecordLink points at. */
function hrefOf(): string {
  return screen.getByRole('link').getAttribute('href') ?? '';
}

describe('RecordLink', () => {
  it('routes to the page owning the referenced record', () => {
    renderLink(<RecordLink recordRef={{ kind: 'deployment', deploymentId: 42 }} />);
    expect(hrefOf().split('?')[0]).toBe('/deployments');
  });

  it('embeds a context that parses back to the original reference', () => {
    const recordRef = { kind: 'job', jobId: 'job-abc-123' } as const;
    renderLink(<RecordLink recordRef={recordRef} />);
    const href = hrefOf();
    expect(parseOperationalContext(href.slice(href.indexOf('?')))?.ref).toEqual(recordRef);
  });

  it('uses the typed record description as its accessible name by default', () => {
    renderLink(<RecordLink recordRef={{ kind: 'job', jobId: 'job-abc-123' }} />);
    expect(screen.getByRole('link', { name: 'Job job-abc-123' })).toBeInTheDocument();
  });

  it('keeps the full identifier in the accessible name when the visible label is truncated', () => {
    renderLink(
      <RecordLink recordRef={{ kind: 'job', jobId: 'job-abc-123' }} label="job-abc…" />,
    );
    // Visible text may be shortened, but assistive tech must still get the full id.
    const link = screen.getByRole('link', { name: 'Job job-abc-123' });
    expect(link).toHaveTextContent('job-abc…');
  });

  it('carries origin and time range so the target can offer an accurate return path', () => {
    renderLink(
      <RecordLink
        recordRef={{ kind: 'incident', incidentId: 7 }}
        origin="/jobs"
        timeRange="7d"
      />,
    );
    const href = hrefOf();
    const parsed = parseOperationalContext(href.slice(href.indexOf('?')));
    expect(parsed?.origin).toBe('/jobs');
    expect(parsed?.timeRange).toBe('7d');
  });

  it('carries view filters through the handoff', () => {
    renderLink(
      <RecordLink
        recordRef={{ kind: 'job', jobId: 'job-1' }}
        filters={{ status: 'failed' }}
      />,
    );
    const href = hrefOf();
    expect(parseOperationalContext(href.slice(href.indexOf('?')))?.filters).toEqual({
      status: 'failed',
    });
  });
});

describe('RecordLink — copy affordance', () => {
  it('has no copy control unless requested', () => {
    renderLink(<RecordLink recordRef={{ kind: 'job', jobId: 'job-1' }} />);
    expect(screen.queryByRole('button', { name: /copy/i })).not.toBeInTheDocument();
  });

  it('copies an absolute link that parses back to the referenced record', async () => {
    // Reads the clipboard userEvent actually wrote to, rather than asserting on
    // a mock, so this fails if the copied href stops resolving to the record.
    const user = userEvent.setup();
    renderLink(<RecordLink recordRef={{ kind: 'job', jobId: 'job-1' }} copyable />);

    // The clipboard write resolves in a microtask after the click, so the
    // resulting state update must be flushed inside act to stay warning-free.
    await act(async () => {
      await user.click(screen.getByRole('button', { name: /copy link to job job-1/i }));
    });

    // Asserts the user-facing confirmation, not just the clipboard side effect.
    await screen.findByRole('button', { name: /link copied/i });

    const copied = await navigator.clipboard.readText();
    expect(copied).toContain('/jobs?');
    expect(parseOperationalContext(copied.slice(copied.indexOf('?')))?.ref).toEqual({
      kind: 'job',
      jobId: 'job-1',
    });
  });
});
