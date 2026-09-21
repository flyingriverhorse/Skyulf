import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test.use({ timezoneId: 'Asia/Kathmandu' });

test('rolling error chart displays the oldest partial hour and the current hour', async ({ page }) => {
  // Real chart rendering receives controlled HTTP counts spanning both edge bins.
  await page.clock.setFixedTime(new Date('2026-09-20T12:30:00Z'));
  await mockBackend(page);
  await page.route('**/api/monitoring/errors?*', route => route.fulfill({ json: {
    total: 1, total_unfiltered: 1, entries: [{
      id: 1, severity: 'critical', status_code: 500, error_type: 'TestError',
      message: 'Timeline event', route: '/test', created_at: '2026-09-20T12:15:00Z',
      resolved_at: null,
    }],
    facets: { severities: [], error_types: [], job_ids: [] }, filters: {},
  } }));
  await page.route('**/api/monitoring/pipeline-logs?*', route => route.fulfill({ json: {
    total: 0, total_unfiltered: 0, entries: [],
    facets: { levels: [], node_types: [], node_ids: [], pipeline_ids: [] }, filters: {},
  } }));
  await page.route('**/api/monitoring/errors/grouped', route => route.fulfill({ json: [] }));
  await page.route('**/api/monitoring/errors/timeline?*', route => route.fulfill({ json: [
    { hour: '2026-09-19T12:00', count: 2 },
    { hour: '2026-09-20T12:00', count: 5 },
  ] }));
  await page.goto('/errors');
  await expect(page.getByText('Errors per hour — last 24 h')).toBeVisible();
  const bars = page.locator('.recharts-bar-rectangle path');
  await expect(bars).toHaveCount(2);
  await bars.first().hover();
  await expect(page.locator('.recharts-tooltip-wrapper')).toContainText('2026-09-19T17:45:00');
  await expect(page.locator('.recharts-tooltip-item-value')).toHaveText('2');
  await bars.last().hover();
  await expect(page.locator('.recharts-tooltip-wrapper')).toContainText('2026-09-20T17:45:00');
  await expect(page.locator('.recharts-tooltip-item-value')).toHaveText('5');
});
