import { expect, test, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

const archive = Buffer.from('UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA==', 'base64');
const olderIdentity = 'Random Forest v2';

async function openRegistry(page: Page): Promise<void> {
  await mockBackend(page);
  const versions = [
    { job_id: 'job-latest', version: 3, status: 'completed', is_deployed: true, deployment_id: 7 },
    { job_id: 'job-older', version: 2, status: 'completed', is_deployed: false },
    { job_id: 'job-failed', version: 1, status: 'failed', is_deployed: false },
  ].map(version => ({
    pipeline_id: 'pipeline-1', node_id: 'model-1', model_type: 'Random Forest',
    source: 'training', metrics: { accuracy: 0.91 }, hyperparameters: {},
    created_at: '2026-10-20T00:00:00Z', artifact_uri: 'model.pkl', ...version,
  }));
  await page.route('**/api/registry/stats', route => route.fulfill({
    json: { total_models: 1, total_versions: 3, active_deployments: 1 },
  }));
  await page.route('**/api/registry/models?*', route => route.fulfill({
    json: [{
      model_type: 'Random Forest', dataset_id: 'iris', dataset_name: 'Iris',
      dataset_type: 'csv', latest_version: versions[0], versions, deployment_count: 1,
    }],
  }));
  await page.goto('/registry');
  const versionsButton = page.getByRole('button', { name: 'View Versions' });
  await versionsButton.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog', { name: 'Random Forest' })).toBeVisible();
}

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
]) {
  test.describe(viewport.name, () => {
    test.use({ viewport });

    test('exports the selected older version in every format with keyboard controls', async ({ page }, testInfo) => {
      // Keyboard selection must reach the selected job and produce a complete browser download.
      await openRegistry(page);
      const requests: { method: string; path: string; mode: string | null }[] = [];
      await page.route('**/api/deployment/export/*', async route => {
        const request = route.request();
        const url = new URL(request.url());
        requests.push({ method: request.method(), path: url.pathname, mode: url.searchParams.get('mode') });
        await route.fulfill({ contentType: 'application/zip', body: archive });
      });
      const select = page.getByRole('combobox', { name: `Project format for ${olderIdentity}` });
      const button = page.getByRole('button', { name: `Export project for ${olderIdentity}` });
      await expect(select).toHaveValue('both');
      await expect(page.getByRole('button', { name: 'Export project for Random Forest v1' })).toHaveCount(0);

      for (const mode of ['both', 'api', 'batch']) {
        await select.focus();
        await page.keyboard.press('Home');
        if (mode === 'api') await page.keyboard.press('ArrowDown');
        if (mode === 'batch') await page.keyboard.press('End');
        await expect(select).toHaveValue(mode);
        await page.keyboard.press('Tab');
        await expect(button).toBeFocused();
        await expect(button).toBeInViewport();
        const downloadPromise = page.waitForEvent('download');
        await page.keyboard.press('Enter');
        const download = await downloadPromise;
        expect(download.suggestedFilename()).toBe('kayra_model_job-older.zip');
        const stream = await download.createReadStream();
        const chunks: Buffer[] = [];
        for await (const chunk of stream) chunks.push(Buffer.from(chunk));
        expect(Buffer.concat(chunks)).toEqual(archive);
      }
      expect(requests).toEqual([
        { method: 'POST', path: '/api/deployment/export/job-older', mode: 'both' },
        { method: 'POST', path: '/api/deployment/export/job-older', mode: 'api' },
        { method: 'POST', path: '/api/deployment/export/job-older', mode: 'batch' },
      ]);
      await page.screenshot({ path: testInfo.outputPath('export-controls.png') });
      await page.keyboard.press('Escape');
      await expect(page.getByRole('dialog', { name: 'Random Forest' })).toHaveCount(0);
      await expect(page.getByRole('button', { name: 'View Versions' })).toBeFocused();
    });

    test('announces the server rejection and permits retry without blocking other versions', async ({ page }, testInfo) => {
      // The backend HTTP exception envelope must retain its message in binary response mode.
      await openRegistry(page);
      let releaseResponse!: () => void;
      const responseGate = new Promise<void>(resolve => { releaseResponse = resolve; });
      let requests = 0;
      await page.route('**/api/deployment/export/job-older?*', async route => {
        requests++;
        if (requests === 1) {
          await responseGate;
          await route.fulfill({
            status: 400,
            json: {
              success: false,
              error: 'HTTP 400',
              message: 'Raw input schema is unavailable. Retrain this model before exporting.',
              status_code: 400,
            },
          });
        } else {
          await route.fulfill({ contentType: 'application/zip', body: archive });
        }
      });
      const button = page.getByRole('button', { name: `Export project for ${olderIdentity}` });
      await button.click();
      await expect(button).toBeDisabled();
      await expect(button).toHaveAttribute('aria-busy', 'true');
      await expect(page.getByRole('combobox', { name: `Project format for ${olderIdentity}` })).toBeDisabled();
      await expect(page.getByRole('button', { name: 'Export project for Random Forest v3' })).toBeEnabled();
      releaseResponse();
      const error = page.getByRole('alert');
      await expect(error).toHaveText('Raw input schema is unavailable. Retrain this model before exporting.');
      await expect(button).toBeEnabled();
      await expect(button).toHaveAccessibleDescription(/Retrain this model/);
      await page.screenshot({ path: testInfo.outputPath('export-error.png') });
      const downloadPromise = page.waitForEvent('download');
      await button.click();
      await downloadPromise;
      await expect(error).toHaveCount(0);
      expect(requests).toBe(2);
    });
  });
}
