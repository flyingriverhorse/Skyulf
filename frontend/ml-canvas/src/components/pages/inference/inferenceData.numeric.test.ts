import { expect, it } from 'vitest';
import { toNumericArray } from './inferenceData';

it('excludes null predictions instead of counting them as zero', () => {
    // Missing predictions must not lower the mean or minimum of the observed population.
    expect(toNumericArray([1, null, 3])).toEqual([1, 3]);
});

it('accepts finite numbers and nonempty numeric strings without coercing other types', () => {
    // Boolean labels, blank values, arrays and objects are not numeric observations.
    expect(toNumericArray([
        0, -2, 1.5, '0', ' 3.5 ', '-1e2', null, undefined, '', '   ', false, true,
        [], [2], {}, { valueOf: () => 7 }, NaN, Infinity, -Infinity, 'NaN', 'Infinity', 'label',
    ])).toEqual([0, -2, 1.5, 0, 3.5, -100]);
});

it('does not coerce values whose numeric conversion can throw', () => {
    // Unsupported values should yield no statistics rather than break result rendering.
    expect(toNumericArray([Symbol('label'), { valueOf: () => { throw new Error('do not coerce'); } }])).toEqual([]);
});
