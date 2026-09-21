import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ColumnDrift, DriftReport } from '../../../core/api/monitoring';
import { exportDriftReportCSV } from './csvExport';

/** Read the real download payload so serialization and browser export stay aligned. */
async function exportedCSV(columns: ColumnDrift[], feature_importances?: Record<string, number>): Promise<string> {
    const report: DriftReport = {
        reference_rows: 100, current_rows: 50, severity: 'warning',
        drifted_columns_count: columns.filter(column => column.drift_detected).length,
        missing_columns: [], new_columns: [], ...(feature_importances ? { feature_importances } : {}),
        column_drifts: Object.fromEntries(columns.map(column => [column.column, column])),
    };
    let download: Blob | undefined;
    vi.stubGlobal('URL', {
        createObjectURL: (blob: Blob) => { download = blob; return 'blob:drift'; },
        revokeObjectURL: vi.fn(),
    });
    exportDriftReportCSV(report, 'Dataset');
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsText(download!);
    });
}

/** A minimal column isolates exported metric values and risk labels. */
function column(name: string, drifted = true): ColumnDrift {
    return { column: name, drift_detected: drifted, suggestions: [], metrics: [] };
}

describe('drift CSV evidence', () => {
    beforeEach(() => vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {}));
    afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

    it('exports categorical and numeric PSI under the same PSI header', async () => {
        /** The downloaded report must retain categorical PSI instead of an empty cell. */
        const numeric = { ...column('numeric'), metrics: [{ metric: 'psi', value: 0, threshold: 0.2, has_drift: false }] };
        const categorical = { ...column('category'), metrics: [{ metric: 'psi_categorical', value: 0.48, threshold: 0.2, has_drift: true }] };
        const csv = await exportedCSV([numeric, categorical]);
        expect(csv).toContain('"numeric","Drifted","","0.000000","","",""');
        expect(csv).toContain('"category","Drifted","","0.480000","","",""');
    });

    it('marks absent importance as unknown and preserves known ranks including zero', async () => {
        /** Missing feature importance must not fabricate risk or treat valid zero as missing. */
        const importance = Object.fromEntries(Array.from({ length: 16 }, (_, index) => [`feature${index}`, 16 - index]));
        importance.zero = 0;
        const names = ['feature0', 'feature5', 'zero', 'missing'];
        const csv = await exportedCSV([...names.map(name => column(name)), column('stable-missing', false)], importance);
        const rows = csv.split('\n');
        expect(rows[1]).toContain('"16.000000","High"');
        expect(rows[2]).toContain('"11.000000","Medium"');
        expect(rows[3]).toContain('"0.000000","Low"');
        expect(rows[4]).toContain('"","Unknown"');
        expect(rows[5]).toContain('"","Unknown"');
    });

    it('escapes quotes while preserving commas and newlines inside quoted cells', async () => {
        /** Feature names containing CSV syntax must remain a single lossless field. */
        const csv = await exportedCSV([column('quoted "name",\nnext line')]);
        expect(csv).toContain('"quoted ""name"",\nnext line","Drifted"');
        expect(csv.split('\n')[0]).not.toContain('Importance');
    });
});
