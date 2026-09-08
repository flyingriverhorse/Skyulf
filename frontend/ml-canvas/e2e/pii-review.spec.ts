import AxeBuilder from '@axe-core/playwright';
import { test, expect, type Page } from '@playwright/test';
import type { EDAReport } from '../src/core/api/eda';
import { mockBackend } from './fixtures/mockApi';

const report: EDAReport = {
  id: 11,
  status: 'COMPLETED',
  profile_data: {
    row_count: 2,
    column_count: 2,
    columns: {
      contact: {
        name: 'contact', dtype: 'Categorical', missing_count: 0, missing_percentage: 0,
        categorical_stats: {
          unique_count: 2, rare_labels_count: 0,
          top_k: [{ value: 'private@example.com', count: 1 }],
        },
      },
      age: { name: 'age', dtype: 'Numeric', missing_count: 1, missing_percentage: 50 },
    },
    sample_data: [{ contact: 'private@example.com', phone: '+1-202-555-0123' }],
    alerts: [
      { type: 'PII', column: 'contact', severity: 'error', message: 'Column contact may contain PII (Email/Phone). private@example.com' },
      { type: 'High Null', column: 'age', severity: 'warning', message: 'Column age has missing values.' },
    ],
  },
};

async function mockProfileDataset(page: Page) {
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({
    json: { sources: [{ id: '101', name: 'Contacts', source_id: 'contacts', type: 'file', created_at: '2026-09-08', rows: 2, columns: 2, format: 'csv' }] },
  }));
  await page.route('**/api/eda/101/history', route => route.fulfill({ json: [] }));
}

test('PII review supports keyboard navigation and both themes without rendering personal values', async ({ page }) => {
  // The dedicated review must omit raw values while generic alerts remain available on Dashboard.
  await mockProfileDataset(page);
  await page.route('**/api/eda/101/latest', route => route.fulfill({ json: report }));
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'light'));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/eda?dataset_id=101');

  await expect(page.getByText('Data Quality Alerts (2)')).toBeVisible();
  await expect(page.getByText(/Column contact may contain PII/)).toBeVisible();
  await expect(page.getByText('Column age has missing values.')).toBeVisible();

  const dashboard = page.getByRole('button', { name: 'Dashboard', exact: true });
  await dashboard.focus();
  await page.keyboard.press('Tab');
  const reviewButton = page.getByRole('button', { name: 'PII Review', exact: true });
  await expect(reviewButton).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(reviewButton).toHaveAttribute('aria-current', 'page');

  const review = page.getByRole('region', { name: 'PII review', exact: true });
  await expect(review.getByRole('rowheader', { name: 'contact' })).toBeVisible();
  await expect(review.getByText('Email / phone', { exact: true })).toBeVisible();
  await expect(review.getByText('Error', { exact: true })).toBeVisible();
  await expect(review.getByText('Advisory heuristics only')).toBeVisible();
  await expect(review).toContainText('does not mask, delete, or block data');
  await expect(page.locator('body')).not.toContainText('private@example.com');
  await expect(page.locator('body')).not.toContainText('+1-202-555-0123');
  expect((await new AxeBuilder({ page }).include('#eda-pii-review').analyze()).violations).toEqual([]);
  await page.screenshot({ path: 'test-results/pii-review-light.png' });

  await page.getByRole('button', { name: 'Switch to dark mode' }).click();
  await expect(page.locator('html')).toHaveClass(/dark/);
  expect((await new AxeBuilder({ page }).include('#eda-pii-review').analyze()).violations).toEqual([]);
  await page.screenshot({ path: 'test-results/pii-review-dark.png' });

  await page.getByRole('button', { name: 'Collapse Sidebar' }).click();
  await reviewButton.focus();
  await page.keyboard.press('Space');
  await expect(review).toBeVisible();
  await dashboard.click();
  await expect(page.getByText(/Column contact may contain PII/)).toBeVisible();
  await expect(page.getByText('Column age has missing values.')).toBeVisible();
});

test('profile loading, missing analysis, failed analysis, and no findings remain distinct', async ({ page }) => {
  // Users must not mistake an unavailable or unfinished profile for a clean PII review.
  await mockProfileDataset(page);
  let releaseReport!: () => void;
  const reportReady = new Promise<void>(resolve => { releaseReport = resolve; });
  await page.route('**/api/eda/101/latest', async route => {
    await reportReady;
    await route.fulfill({ json: report });
  });
  await page.goto('/eda?dataset_id=101');
  await expect(page.getByText('Analyzing dataset...')).toBeVisible();
  await expect(page.getByText('No PII findings recorded')).toHaveCount(0);
  releaseReport();
  await expect(page.getByRole('button', { name: 'PII Review' })).toBeVisible();

  const states = [
    { status: 404, json: { detail: 'No report' }, text: 'No analysis found for this dataset.' },
    { status: 200, json: { id: 11, status: 'PENDING' }, text: 'Analysis in progress...' },
    { status: 200, json: { id: 11, status: 'FAILED', error_message: 'Profiling failed' }, text: 'Analysis Failed' },
    { status: 500, json: { detail: 'Unavailable' }, text: 'Failed to load report' },
  ];
  for (const state of states) {
    await page.route('**/api/eda/101/latest', route => route.fulfill({ status: state.status, json: state.json }));
    await page.reload();
    await expect(page.getByText(state.text, { exact: true })).toBeVisible({ timeout: 20_000 });
    await expect(page.getByText('No PII findings recorded')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'PII Review' })).toHaveCount(0);
  }

  await page.route('**/api/eda/101/latest', route => route.fulfill({
    json: { id: 12, status: 'COMPLETED', profile_data: { row_count: 2, column_count: 0, columns: {} } },
  }));
  await page.reload();
  await page.getByRole('button', { name: 'PII Review' }).click();
  await expect(page.getByText('No PII findings recorded')).toBeVisible();
  await expect(page.getByRole('region', { name: 'PII review', exact: true })).toContainText('does not guarantee');
});
