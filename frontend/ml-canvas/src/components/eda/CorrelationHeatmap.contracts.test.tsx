import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { CorrelationHeatmap } from './CorrelationHeatmap';
import { CorrelationsTab } from './tabs/CorrelationsTab';

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it('preserves row labels and CSV values when real column names resemble internal keys', async () => {
  // Neither the row-label cell nor React column identity may collide with user names.
  const columns = ['variable', 'row_label', 'column_0', '__proto__'];
  const values = columns.map((_, i) => columns.map((_, j) => i === j ? 1 : 0));
  const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
  let blob!: Blob;
  vi.stubGlobal('URL', { createObjectURL: (value: Blob) => { blob = value; return 'blob:csv'; }, revokeObjectURL: vi.fn() });
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  render(<CorrelationHeatmap data={{ columns, values }} />);
  fireEvent.click(screen.getByRole('button', { name: 'View data table' }));
  const table = screen.getByRole('table');
  const rows = within(table).getAllByRole('row').slice(1);
  expect(rows.map(row => within(row).getAllByRole('cell')[0]!.textContent)).toEqual(columns);
  fireEvent.click(screen.getByRole('button', { name: 'Download CSV' }));
  const csv = await new Promise<string>(resolve => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.readAsText(blob);
  });
  expect(csv).toBe('Variable,variable,row_label,column_0,__proto__\nvariable,1,0,0,0\nrow_label,0,1,0,0\ncolumn_0,0,0,1,0\n__proto__,0,0,0,1');
  expect(errors).not.toHaveBeenCalled();
});

it('keeps zero neutral, missing distinct and small correlations distinguishable', () => {
  // Positive/negative weak correlations must not collapse into a fixed-opacity block.
  const values = [-1, -0.4, -0.02, 0, 0.02, 0.4, 1, null];
  render(<CorrelationHeatmap data={{ columns: values.map((_, i) => `c${i}`), values: [values] }} />);
  const cells = values.map((value, index) => screen.getByTitle(`c0 vs c${index}: ${value === null ? 'N/A' : value.toFixed(3)}`));
  expect(cells[3]).toHaveStyle({ backgroundColor: 'rgb(255, 255, 255)' });
  expect(new Set(cells.map(cell => cell.style.backgroundColor)).size).toBe(8);
  expect(screen.getByText('Missing', { exact: true })).toBeVisible();
});

it('exports the same cell colors and includes both visual and analysis omissions with a legend', () => {
  // PNGs must explain their partial matrix without relying on an adjacent, collapsed table.
  const sample = [-1, -0.4, -0.02, 0, 0.02, 0.4, 1, null];
  const columns = Array.from({ length: 22 }, (_, index) => `column${index}`);
  const values = columns.map(() => columns.map((_, index) => sample[index % sample.length]!));
  const fills: Array<{ color: string; width: number; height: number }> = [];
  const text: string[] = [];
  const ctx = {
    fillStyle: '', font: '', textAlign: '', textBaseline: '',
    measureText: (value: string) => ({ width: value.length * 6 }),
    fillRect: (_x: number, _y: number, width: number, height: number) => { fills.push({ color: ctx.fillStyle, width, height }); },
    fillText: (value: string) => text.push(value), save: vi.fn(), restore: vi.fn(), translate: vi.fn(), rotate: vi.fn(),
  };
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/png;base64,test');
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  render(<CorrelationsTab profile={{ correlations: { columns, values, total_columns: 24, omitted_columns: ['omitted_a', 'omitted_b'] } }} />);
  fireEvent.click(screen.getByRole('button', { name: 'Download Matrix' }));
  const cellColors = fills.filter(fill => fill.width === 58 && fill.height === 58).slice(0, 8).map(fill => fill.color);
  expect(cellColors).toEqual(sample.map((value, index) => screen.getByTitle(`column0 vs column${index}: ${value === null ? 'N/A' : value.toFixed(3)}`).style.backgroundColor));
  expect(text.join(' ')).toMatch(/first 20 of 22 columns/i);
  expect(text.join(' ')).toContain('omitted_a');
  expect(text.join(' ')).toContain('Missing');
  expect(screen.getByText(/first 20 of 22 columns/i)).toBeVisible();
  expect(screen.getByText(/omitted_a/)).toBeVisible();
});
