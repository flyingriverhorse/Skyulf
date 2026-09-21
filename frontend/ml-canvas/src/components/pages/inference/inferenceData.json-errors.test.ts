import { expect, it } from 'vitest';
import { analyseInput, extractJsonErrorPosition } from './inferenceData';

it('preserves the native V8 trailing-comma position at line 1 column 8', () => {
  /** A known valid position must not regress while supporting coordinate-free errors. */
  expect(analyseInput('{"a":1,}')).toMatchObject({ valid: false, rows: 0, line: 1, column: 8 });
});

it('retains the native explanation without inventing coordinates', () => {
  /** Invalid literals may have no position in V8 and must still explain the parse failure. */
  let message = '';
  try { JSON.parse('undefined'); } catch (error) { message = (error as Error).message; }
  expect(message).not.toBe('');
  expect(extractJsonErrorPosition(message, 'undefined')).toBeNull();
  expect(analyseInput('undefined')).toEqual({ valid: false, rows: 0, message });
});

it('supports Firefox coordinates and rejects messages without them', () => {
  /** Cross-browser messages may provide an explicit line/column or only an explanation. */
  expect(extractJsonErrorPosition('JSON.parse: unexpected character at line 3 column 5 of the JSON data', ''))
    .toEqual({ line: 3, column: 5 });
  expect(extractJsonErrorPosition('Unexpected end of JSON input', '[')).toBeNull();
});
