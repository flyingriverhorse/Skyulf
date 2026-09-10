import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { JobLogs } from './JobLogs';
import type { JobInfo } from '../../../../core/api/jobs';

function renderLog(message: string) {
  const job: JobInfo = {
    job_id: 'log-test', pipeline_id: 'pipeline', node_id: 'model', job_type: 'training',
    status: 'completed', start_time: null, end_time: null, error: null, result: null,
    created_at: '2026-09-10T10:00:00Z', logs: [message],
  };
  render(<JobLogs job={job} controls={{
    autoScroll: false, setAutoScroll: vi.fn(), wrapLines: true, setWrapLines: vi.fn(),
    copied: false, logsEndRef: React.createRef<HTMLDivElement>(), handleCopyLogs: vi.fn(),
  }} />);
  return screen.getAllByRole('cell')[1]!;
}

describe('JobLogs numeric highlighting', () => {
  it.each([
    ['3.14s', '3.14s', 'text-teal-400'],
    ['250ms', '250ms', 'text-teal-400'],
    ['-0.05s', '-0.05s', 'text-teal-400'],
    ['1.e-3 s', '1.e-3 s', 'text-teal-400'],
    ['12\tms', '12\tms', 'text-teal-400'],
    ['12\u00a0s', '12\u00a0s', 'text-teal-400'],
    ['98.5%', '98.5%', 'text-emerald-400'],
    ['-12.%', '-12.%', 'text-emerald-400'],
    ['-0.25E+4', '-0.25E+4', 'text-emerald-400'],
    ['run123.45', '123.45', 'text-emerald-400'],
    ['run1-2s', '-2s', 'text-teal-400'],
    ['run1-2%', '-2%', 'text-emerald-400'],
    ['run1-2.5', '-2.5', 'text-emerald-400'],
    ['1e+2%', '2%', 'text-emerald-400'],
    ['1.25e+', '1.25', 'text-emerald-400'],
  ])('preserves numeric token styling in %j', (message, token, cls) => {
    // Signs, malformed suffixes and embedded numbers must keep their existing highlighting.
    const cell = renderLog(message);
    const spans = Array.from(cell.querySelectorAll('span'), span => [span.textContent, span.className]);
    expect(cell.textContent).toBe(message);
    expect(spans).toEqual([[token, cls]]);
  });

  it('leaves a number and unit embedded in a word unstyled', () => {
    // A duration unit must still end at a word boundary.
    const cell = renderLog('123msX');
    expect(cell.textContent).toBe('123msX');
    expect(cell.querySelectorAll('span')).toHaveLength(0);
  });

  it('keeps earlier rules ahead of numeric highlighting', () => {
    // Metrics, progress and quoted text retain their complete token boundaries and colors.
    const message = 'loss=-0.25 Epoch 2/5 "98.5%" 3.14s';
    const cell = renderLog(message);
    expect(cell.textContent).toBe(message);
    expect(Array.from(cell.querySelectorAll('span'), span => [span.textContent, span.className])).toEqual([
      ['loss', 'text-sky-400'], ['=', 'text-gray-600'], ['-0.25', 'text-emerald-400'],
      ['Epoch ', 'text-orange-400'], ['2', 'text-orange-300 font-semibold'],
      ['/', 'text-gray-500'], ['5', 'text-orange-300'],
      ['"98.5%"', 'text-amber-300/80'], ['3.14s', 'text-teal-400'],
    ]);
  });

  it('renders long numeric runs with no duration, percentage or decimal suffix', () => {
    // Missing suffixes must leave the entire integer visible without splitting its digits.
    const digits = '7'.repeat(4000);
    const cell = renderLog(`${digits}!`);
    expect(cell.textContent).toBe(`${digits}!`);
    expect(cell.querySelector('span')).toHaveTextContent(digits);
    expect(cell.querySelectorAll('span')).toHaveLength(1);
  });
});
