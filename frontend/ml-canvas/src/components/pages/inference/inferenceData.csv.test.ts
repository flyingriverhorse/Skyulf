import { describe, expect, it } from 'vitest';
import { parseCsv, rowsToCsv } from './inferenceData';

describe('inference CSV roundtrips', () => {
    it('accepts a UTF-8 BOM before a quoted header', () => {
        // Spreadsheet exports may prefix a BOM before the first CSV field.
        expect(parseCsv('\uFEFF"a,b",c\r\n"x,y",1')).toEqual([{ 'a,b': 'x,y', c: 1 }]);
    });

    it.each(['\n', '\r\n'])('parses quoted fields with %j record separators', separator => {
        // Quoted delimiters and embedded line breaks must not shift feature values.
        const csv = ['"feature,name","say""what",notes,empty,last',
            '"a,b","say ""hello""","first\r\nsecond\nthird",,3'].join(separator);
        expect(parseCsv(csv)).toEqual([{
            'feature,name': 'a,b', 'say"what': 'say "hello"',
            notes: 'first\r\nsecond\nthird', empty: '', last: 3,
        }]);
    });

    it('exports and reimports escaped headers, whitespace, and carriage returns', () => {
        // Headers use the same escaping rules as cells, including lone carriage returns.
        const rows = [{ 'header,one': 'a,b', 'say"what': 'a"b', 'line\nbreak': 'a\nb',
            'carriage\rreturn': 'a\rb', ' padded ': ' padded ', empty: '' }];
        expect(parseCsv(rowsToCsv(rows))).toEqual(rows);
    });

    it('retains an empty value in a single-column export', () => {
        // An empty record must not disappear as an ignorable blank physical line.
        expect(parseCsv(rowsToCsv([{ value: '' }]))).toEqual([{ value: '' }]);
    });

    it('preserves existing numeric conversion, including quoted numeric cells', () => {
        // CSV quoting protects delimiters; it does not change the numeric inference policy.
        expect(parseCsv('a,b,c,d,e,f,g\n1.0,1e5,-.5,001,+1,1e999,"2.5"')).toEqual([
            { a: 1, b: 100000, c: -0.5, d: 1, e: '+1', f: '1e999', g: 2.5 },
        ]);
    });

    it('ignores blank lines and trims unquoted cells', () => {
        // Existing plain CSV files keep their whitespace and trailing-line behavior.
        expect(parseCsv('\n a , b \r\n 1 , text \r\n\n')).toEqual([{ a: 1, b: 'text' }]);
    });

    it.each(['a,b\n1', 'a,b\n1,2,3'])('rejects mismatched widths: %s', csv => {
        // Missing or excess cells must not silently produce a different inference payload.
        expect(() => parseCsv(csv)).toThrow(/CSV row 2.*expected 2.*received/i);
    });

    it.each(['a\n"unfinished', 'a\n"closed"junk', 'a\nun"quoted'])('rejects malformed quoting: %s', csv => {
        // An invalid quote cannot be treated as valid prediction input.
        expect(() => parseCsv(csv)).toThrow(/CSV row 2.*quot/i);
    });

    it('rejects duplicate headers instead of overwriting a feature', () => {
        // Object conversion cannot represent two distinct columns sharing a header.
        expect(() => parseCsv('a,a\n1,2')).toThrow(/duplicate.*header/i);
    });

    it('preserves special object-property names as own features', () => {
        // A CSV header must not invoke the legacy prototype setter.
        const rows = parseCsv('__proto__,constructor\nvalue,other');
        expect(Object.keys(rows[0] ?? {})).toEqual(['__proto__', 'constructor']);
        expect(JSON.stringify(rows)).toBe('[{"__proto__":"value","constructor":"other"}]');
    });
});
