import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { OverviewCards } from './OverviewCards';

describe('OverviewCards duplicate counts', () => {
    it.each([
        [null, 'Unavailable'],
        [undefined, 'Unavailable'],
        [0, '0'],
        [3, '3'],
    ] as const)('shows %s as %s', (duplicateRows, expected) => {
        render(<OverviewCards profile={{
            row_count: 10,
            column_count: 2,
            columns: {},
            ...(duplicateRows === undefined ? {} : { duplicate_rows: duplicateRows }),
        }} />);

        const card = screen.getByText('Duplicates').parentElement;
        expect(card).not.toBeNull();
        expect(within(card!).getByText(expected)).toBeInTheDocument();
    });
});
