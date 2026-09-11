import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';
import type { Page } from '@playwright/test';
import type { SavedThresholdInfo, ThresholdPreviewResult } from '../src/core/api/thresholdTuning';

/**
 * Visual verification of the F-13 "Tune decision threshold" checkbox:
 * shown for classification nodes in Advanced (tuning) mode, hidden for
 * regression, off by default, and toggleable.
 */

const HYPERPARAMETER_DEFS = [
  {
    name: 'n_estimators',
    label: 'Number of Trees',
    type: 'number',
    default: 100,
    min: 10,
    max: 1000,
    tunable: true,
  },
];

const addNode = async (page: import('@playwright/test').Page, label: string) => {
  await page.locator('aside [draggable="true"]').filter({ hasText: label }).first().click();
  await expect(page.locator('.react-flow__node')).toHaveCount(1);
};

test.describe('Decision threshold tuning control', () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
    // Registered after the catch-all so these take precedence (Playwright
    // matches routes in reverse registration order).
    await page.route('**/api/pipeline/hyperparameters/*', (route) =>
      route.fulfill({ json: HYPERPARAMETER_DEFS })
    );
    await page.route('**/api/pipeline/hyperparameters/*/defaults*', (route) =>
      route.fulfill({ json: { n_estimators: [50, 100, 200] } })
    );
  });

  test('classification: advanced mode shows the checkbox, off by default, toggleable', async ({ page }) => {
    await page.goto('/canvas');
    await expect(page.locator('.react-flow')).toBeVisible({ timeout: 10_000 });
    await addNode(page, 'Classification');

    await page.getByRole('button', { name: 'Advanced (Tuning)' }).click();
    await expect(page.getByText('Tuning Strategy')).toBeVisible();

    const checkbox = page.getByRole('checkbox', { name: 'Tune decision threshold', exact: true });
    await expect(checkbox).toBeVisible();
    await expect(checkbox).not.toBeChecked();

    await page.getByText('Tune decision threshold').click();
    await expect(checkbox).toBeChecked();

    await page.screenshot({ path: 'test-results/threshold-tuning-classification.png', fullPage: true });
  });

  test('regression: advanced mode does not show the checkbox', async ({ page }) => {
    await page.goto('/canvas');
    await expect(page.locator('.react-flow')).toBeVisible({ timeout: 10_000 });
    await addNode(page, 'Regression');

    await page.getByRole('button', { name: 'Advanced (Tuning)' }).click();
    await expect(page.getByText('Tuning Strategy')).toBeVisible();

    await expect(page.getByRole('checkbox', { name: 'Tune decision threshold', exact: true })).toHaveCount(0);
  });
});

const PREVIEW_THRESHOLDS: ThresholdPreviewResult = {
  thresholds: { '0': 0.4, '1': 0.6 }, classes: [0, 1], metric: 'f1', split_used: 'validation',
};

const emptySavedThresholds = (): SavedThresholdInfo => ({
  thresholds: null, classes: null, metric: null, split_used: null,
  computed_at: null, source: null, enabled: false,
});

async function mockThresholdLifecycle(page: Page) {
  await mockBackend(page);
  const saved = new Map<string, SavedThresholdInfo>();
  const mutations: string[] = [];
  const jobs = ['threshold-a', 'threshold-b'].map((jobId, index) => ({
    job_id: jobId, pipeline_id: `threshold${index + 1}`, node_id: 'classifier',
    job_type: 'training', status: 'completed', model_type: 'random_forest_classifier',
    model_family: 'classification', dataset_name: `Threshold dataset ${index + 1}`,
    start_time: '2026-09-09T10:00:00Z', end_time: '2026-09-09T10:01:00Z',
    created_at: '2026-09-09T10:00:00Z', error: null, result: {}, metrics: { test_accuracy: 0.75 },
  }));
  await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: jobs }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  const split = {
    y_true: [0, 0, 1, 1], y_pred: [0, 1, 1, 1],
    y_proba: { classes: [0, 1], values: [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]] },
  };
  await page.route('**/api/pipeline/jobs/*/evaluation', route => route.fulfill({
    json: { problem_type: 'classification', splits: { train: split, validation: split } },
  }));
  await page.route('**/api/pipeline/jobs/*/thresholds**', async route => {
    const request = route.request();
    const match = new URL(request.url()).pathname.match(/\/jobs\/([^/]+)\/thresholds(?:\/(.*))?$/)!;
    const [, jobId, action] = match;
    const current = saved.get(jobId!) ?? emptySavedThresholds();
    if (request.method() === 'GET') {
      await route.fulfill({ json: current });
      return;
    }
    mutations.push(`${jobId}:${action ?? 'clear'}`);
    if (action === 'preview') {
      expect(request.postDataJSON()).toEqual({ metric: 'f1' });
      await route.fulfill({ json: PREVIEW_THRESHOLDS });
    } else if (action === 'save') {
      expect(request.postDataJSON()).toEqual(PREVIEW_THRESHOLDS);
      saved.set(jobId!, { ...PREVIEW_THRESHOLDS, source: null, computed_at: '2026-09-09T10:02:00Z', enabled: true });
      await route.fulfill({ json: { status: 'saved' } });
    } else if (action === 'toggle') {
      if (!current.thresholds) {
        await route.fulfill({ status: 400, json: { detail: 'Job has no saved tuned thresholds to toggle.' } });
        return;
      }
      const { enabled } = request.postDataJSON() as { enabled: boolean };
      saved.set(jobId!, { ...current, enabled });
      await route.fulfill({ json: { status: 'toggled', enabled } });
    } else {
      expect(request.method()).toBe('DELETE');
      saved.delete(jobId!);
      await route.fulfill({ json: { status: 'cleared' } });
    }
  });
  return { saved, mutations };
}

async function openThresholdEvaluation(page: Page) {
  await page.goto('/canvas');
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click();
  await page.getByText('Threshold dataset 1', { exact: false }).click();
  if ((page.viewportSize()?.width ?? 1440) < 768) {
    await page.getByTitle('Collapse Sidebar').click();
  }
  await page.getByRole('button', { name: 'Model Evaluation', exact: true }).click();
  await page.getByRole('button', { name: 'Threshold Tuning', exact: true }).click();
}

test('evaluation distinguishes preview from saved thresholds through save, toggle, reload and clear', async ({ page }) => {
  // Real page wiring must match the server: Preview is temporary; Save persists and enables.
  const { saved, mutations } = await mockThresholdLifecycle(page);
  await openThresholdEvaluation(page);
  const toggle = page.getByRole('checkbox', { name: 'Use tuned thresholds at prediction time' });
  await expect(toggle).toBeDisabled();
  await expect(toggle).not.toBeChecked();

  await page.getByRole('button', { name: 'Preview', exact: true }).click();
  const save = page.getByRole('button', { name: 'Save', exact: true });
  await expect(save).toBeVisible();
  await expect(toggle).toBeDisabled();
  expect(saved.has('threshold-a')).toBe(false);
  expect(mutations).toEqual(['threshold-a:preview']);

  await save.click();
  await expect(toggle).toBeEnabled();
  await expect(toggle).toBeChecked();
  expect(saved.get('threshold-a')?.enabled).toBe(true);
  await toggle.click();
  await expect(toggle).not.toBeChecked();
  await expect(toggle).toBeEnabled();
  expect(saved.get('threshold-a')?.enabled).toBe(false);
  await toggle.click();
  await expect(toggle).toBeChecked();
  await expect(toggle).toBeEnabled();
  expect(saved.get('threshold-a')?.enabled).toBe(true);

  await page.getByText('Threshold dataset 2', { exact: false }).click();
  await page.getByRole('tab', { name: 'threshold2', exact: true }).click();
  await expect(toggle).toBeDisabled();
  await expect(toggle).not.toBeChecked();
  await page.getByRole('tab', { name: 'threshold1', exact: true }).click();
  await expect(toggle).toBeEnabled();
  await expect(toggle).toBeChecked();

  await openThresholdEvaluation(page);
  await expect(toggle).toBeEnabled();
  await expect(toggle).toBeChecked();
  await page.getByRole('button', { name: 'Clear', exact: true }).click();
  await expect(toggle).toBeDisabled();
  await expect(toggle).not.toBeChecked();
  expect(saved.has('threshold-a')).toBe(false);
  expect(mutations).toEqual([
    'threshold-a:preview', 'threshold-a:save', 'threshold-a:toggle', 'threshold-a:toggle', 'threshold-a:clear',
  ]);
});

for (const width of [1440, 390]) {
  test(`legacy ROC AUC thresholds preserve use and allow a supported preview at ${width}px`, async ({ page }) => {
    // Saved cutoffs stay usable, while keyboard selection submits a real threshold objective.
    await page.setViewportSize({ width, height: 1000 });
    const { saved } = await mockThresholdLifecycle(page);
    saved.set('threshold-a', {
      ...PREVIEW_THRESHOLDS, metric: 'roc_auc', source: null,
      computed_at: '2026-09-09T10:02:00Z', enabled: true,
    });
    let requestedMetric: string | undefined;
    await page.route('**/api/pipeline/jobs/threshold-a/thresholds/preview', async route => {
      requestedMetric = (route.request().postDataJSON() as { metric: string }).metric;
      await route.fulfill({ json: { ...PREVIEW_THRESHOLDS, metric: 'balanced_accuracy' } });
    });
    await openThresholdEvaluation(page);
    const metric = page.getByRole('combobox', { name: 'Threshold tuning metric' });
    await expect(metric).toHaveValue('f1');
    await expect(metric.getByRole('option', { name: 'ROC AUC' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled();
    await expect(page.getByText(/click Preview before saving a replacement/)).toBeVisible();
    const toggle = page.getByRole('checkbox', { name: 'Use tuned thresholds at prediction time' });
    await expect(toggle).toBeChecked();
    await toggle.click();
    await expect(toggle).not.toBeChecked();
    await expect(toggle).toBeEnabled();
    await toggle.click();
    await expect(toggle).toBeChecked();
    await expect(toggle).toBeEnabled();
    await metric.focus();
    await page.keyboard.press('End');
    await page.keyboard.press('Enter');
    await expect(metric).toHaveValue('balanced_accuracy');
    await page.getByRole('button', { name: 'Preview', exact: true }).click();
    await expect.poll(() => requestedMetric).toBe('balanced_accuracy');
    await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeEnabled();
    expect(saved.get('threshold-a')?.metric).toBe('roc_auc');
    expect(saved.get('threshold-a')?.enabled).toBe(true);
  });
}
