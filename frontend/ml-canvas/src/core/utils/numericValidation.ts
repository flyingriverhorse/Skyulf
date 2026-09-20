import type { ValidationResult } from '../types/nodes';

/** Preserve an invalid draft as NaN so validation blocks Run instead of truncating it. */
export function numericDraft(value: string): number {
  return value.trim() === '' ? Number.NaN : Number(value);
}

/** Native number controls display invalid drafts as empty without React NaN warnings. */
export function numericInputValue(value: unknown, fallback: number): number | '' {
  return value === undefined ? fallback : typeof value === 'number' && Number.isFinite(value) ? value : '';
}

/** Validate only supplied values; missing settings retain their documented defaults. */
export function numericIssue(field: string, value: unknown, min: number, integer = false, max = Infinity): ValidationResult | undefined {
  if (value === undefined) return;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) {
    return { isValid: false, field, message: `${field} must be a finite ${integer ? 'integer' : 'number'} between ${min} and ${max === Infinity ? 'the supported maximum' : max}.` };
  }
}

/** Check the numeric controls actually consumed by this training mode. */
export function modelNumericIssue(config: object): ValidationResult | undefined {
  const data = config as Record<string, unknown>;
  const checks: (ValidationResult | undefined)[] = [];
  if (data.run_mode === 'advanced') checks.push(numericIssue('n_trials', data.n_trials, 1, true));
  if (data.cv_enabled !== false) checks.push(numericIssue('cv_folds', data.cv_folds, 2, true));
  if (data.strategy === 'stacking') checks.push(numericIssue('cv', data.cv, 2, true));
  if (data.task === 'classification' && data.calibrate_base_models === true) checks.push(numericIssue('calibration_cv', data.calibration_cv, 2, true));
  if (data.run_mode !== 'advanced' && data.n_jobs != null && (typeof data.n_jobs !== 'number' || !Number.isSafeInteger(data.n_jobs) || data.n_jobs === 0)) {
    checks.push({ isValid: false, field: 'n_jobs', message: 'Parallel jobs must be a nonzero integer (-1 uses all cores).' });
  }
  for (const field of ['random_state', 'cv_random_state']) {
    if (data[field] !== undefined && data[field] !== null) checks.push(numericIssue(field, data[field], 0, true, 4294967295));
  }
  if (data.run_mode === 'advanced' && (containsInvalidCandidates(data.search_space) || Object.keys((data.invalid_search_space ?? {}) as object).length > 0)) {
    checks.push({ isValid: false, field: 'search_space', message: 'Search parameters need at least one candidate and numeric candidates must be finite.' });
  }
  return checks.find(Boolean);
}

/** Null candidates and categorical strings retain their original search semantics. */
function containsInvalidCandidates(value: unknown): boolean {
  if (typeof value === 'number') return !Number.isFinite(value);
  if (Array.isArray(value)) return value.length === 0 || value.some(containsInvalidCandidates);
  return value != null && typeof value === 'object' && Object.values(value).some(containsInvalidCandidates);
}

/** sklearn accepts finite numeric thresholds or mean/median expressions. */
export function modelThresholdValid(value: unknown): boolean {
  if (value == null) return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value !== 'string') return false;
  const text = value.trim();
  if (text !== '' && Number.isFinite(Number(text))) return true;
  if (/^(mean|median)$/.test(text)) return true;
  const expression = /^(.+)\*\s*(mean|median)$/.exec(text);
  return expression !== null && expression[1]!.trim() !== '' && Number.isFinite(Number(expression[1]));
}
